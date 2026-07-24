## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0013


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9893196, 0.9936289, 0.9893196, 0.9936289, -0.0029459, 0.0029459)
1: (-0.0039252, -0.0028515, -0.0039252, -0.0028515, -0.0007340, 0.0007340)
2: (0.0050574, 0.0107476, 0.0050574, 0.0107476, -0.0038901, 0.0038901)
3: (-0.0061650, -0.0035750, -0.0061650, -0.0035750, -0.0017706, 0.0017706)
4: (0.0015067, 0.0026081, 0.0015067, 0.0026081, -0.0007529, 0.0007529)
5: (0.0053204, 0.0124771, 0.0053204, 0.0124771, -0.0048927, 0.0048927)
6: (-0.0016260, 0.0001905, -0.0016260, 0.0001905, -0.0012418, 0.0012418)
7: (-0.0073446, -0.0026448, -0.0073446, -0.0026448, -0.0032130, 0.0032130)
8: (-0.0034266, -0.0009550, -0.0034266, -0.0009550, -0.0016897, 0.0016897)
9: (-0.0007564, 0.0021094, -0.0007564, 0.0021094, -0.0019592, 0.0019592)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.65 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0017064, upper bound: 0.0017064

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016616, upper bound: 0.0015977
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016616, upper bound: 0.0016617
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -0.0016616, upper bound: 0.0015977
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -0.0016616, upper bound: 0.0016617

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9893218, 0.9934399, 0.9893202, 0.9935727, -0.0028521, 0.0027351
1: -0.0039247, -0.0028986, -0.0039251, -0.0028654, -0.0007107, 0.0006815
2: 0.0053068, 0.0107447, 0.0051314, 0.0107468, -0.0036117, 0.0037662
3: -0.0061636, -0.0036885, -0.0061646, -0.0036087, -0.0017142, 0.0016439
4: 0.0015550, 0.0026075, 0.0015211, 0.0026079, -0.0006990, 0.0007289
5: 0.0056340, 0.0124734, 0.0054134, 0.0124760, -0.0045425, 0.0047369
6: -0.0016250, 0.0001109, -0.0016257, 0.0001669, -0.0012023, 0.0011529
7: -0.0073421, -0.0028508, -0.0073439, -0.0027059, -0.0031106, 0.0029830
8: -0.0034253, -0.0010634, -0.0034262, -0.0009872, -0.0016359, 0.0015687
9: -0.0006308, 0.0021079, -0.0007192, 0.0021090, -0.0018190, 0.0018969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015079
time: 0.75 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015780, upper bound: 0.0015078
time: 0.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9892282, 0.9935197, 0.9893202, 0.9935927, -0.0029489, 0.0028184
1: -0.0039480, -0.0028787, -0.0039251, -0.0028605, -0.0007348, 0.0007023
2: 0.0052015, 0.0108682, 0.0051051, 0.0107469, -0.0037216, 0.0038939
3: -0.0062199, -0.0036406, -0.0061646, -0.0035967, -0.0017724, 0.0016939
4: 0.0015346, 0.0026314, 0.0015160, 0.0026079, -0.0007203, 0.0007537
5: 0.0055016, 0.0126288, 0.0053803, 0.0124762, -0.0046808, 0.0048975
6: -0.0016645, 0.0001445, -0.0016258, 0.0001753, -0.0012431, 0.0011880
7: -0.0074442, -0.0027638, -0.0073440, -0.0026842, -0.0032161, 0.0030738
8: -0.0034790, -0.0010176, -0.0034263, -0.0009757, -0.0016913, 0.0016165
9: -0.0006839, 0.0021702, -0.0007324, 0.0021091, -0.0018744, 0.0019612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015780
time: 0.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015780, upper bound: 0.0015780
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015079
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -0.0015780, upper bound: 0.0015078
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015780
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 0, lower bound: -0.0015780, upper bound: 0.0015780

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9893218, 0.9934399, 0.9893228, 0.9934980, -0.0027672, 0.0027248
1: -0.0039247, -0.0028986, -0.0039244, -0.0028841, -0.0006895, 0.0006790
2: 0.0053068, 0.0107447, 0.0052301, 0.0107435, -0.0035981, 0.0036541
3: -0.0061636, -0.0036885, -0.0061631, -0.0036536, -0.0016632, 0.0016377
4: 0.0015550, 0.0026075, 0.0015402, 0.0026073, -0.0006964, 0.0007072
5: 0.0056340, 0.0124734, 0.0055375, 0.0124719, -0.0045255, 0.0045959
6: -0.0016250, 0.0001109, -0.0016247, 0.0001353, -0.0011665, 0.0011486
7: -0.0073421, -0.0028508, -0.0073412, -0.0027874, -0.0030180, 0.0029718
8: -0.0034253, -0.0010634, -0.0034248, -0.0010300, -0.0015872, 0.0015629
9: -0.0006308, 0.0021079, -0.0006695, 0.0021074, -0.0018122, 0.0018404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0014910
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015078
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9893244, 0.9933678, 0.9890288, 0.9933963, -0.0028682, 0.0029709
1: -0.0039240, -0.0029165, -0.0039977, -0.0029094, -0.0007147, 0.0007403
2: 0.0054020, 0.0107413, 0.0053644, 0.0111316, -0.0039230, 0.0037875
3: -0.0061621, -0.0037319, -0.0063398, -0.0037147, -0.0017239, 0.0017856
4: 0.0015734, 0.0026069, 0.0015661, 0.0026824, -0.0007593, 0.0007331
5: 0.0057538, 0.0124692, 0.0057064, 0.0129601, -0.0049342, 0.0047636
6: -0.0016240, 0.0000805, -0.0017486, 0.0000925, -0.0012091, 0.0012523
7: -0.0073394, -0.0029295, -0.0076617, -0.0028983, -0.0031282, 0.0032402
8: -0.0034239, -0.0011047, -0.0035934, -0.0010884, -0.0016451, 0.0017040
9: -0.0005829, 0.0021063, -0.0006019, 0.0023029, -0.0019759, 0.0019076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0014420
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0014481
time: 0.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9892282, 0.9935197, 0.9893227, 0.9935172, -0.0028611, 0.0028081
1: -0.0039480, -0.0028787, -0.0039245, -0.0028793, -0.0007129, 0.0006997
2: 0.0052015, 0.0108682, 0.0052046, 0.0107436, -0.0037081, 0.0037780
3: -0.0062199, -0.0036406, -0.0061631, -0.0036420, -0.0017196, 0.0016878
4: 0.0015346, 0.0026314, 0.0015352, 0.0026073, -0.0007177, 0.0007312
5: 0.0055016, 0.0126288, 0.0055055, 0.0124721, -0.0046638, 0.0047518
6: -0.0016645, 0.0001445, -0.0016247, 0.0001435, -0.0012061, 0.0011837
7: -0.0074442, -0.0027638, -0.0073412, -0.0027664, -0.0031204, 0.0030627
8: -0.0034790, -0.0010176, -0.0034248, -0.0010190, -0.0016410, 0.0016106
9: -0.0006839, 0.0021702, -0.0006823, 0.0021074, -0.0018676, 0.0019028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015532
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015780
time: 0.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9892308, 0.9934531, 0.9890288, 0.9934200, -0.0029618, 0.0030538
1: -0.0039473, -0.0028953, -0.0039977, -0.0029035, -0.0007380, 0.0007609
2: 0.0052894, 0.0108648, 0.0053331, 0.0111317, -0.0040325, 0.0039110
3: -0.0062183, -0.0036806, -0.0063398, -0.0037005, -0.0017801, 0.0018354
4: 0.0015516, 0.0026308, 0.0015601, 0.0026824, -0.0007805, 0.0007570
5: 0.0056122, 0.0126246, 0.0056671, 0.0129602, -0.0050719, 0.0049190
6: -0.0016634, 0.0001164, -0.0017486, 0.0001025, -0.0012485, 0.0012873
7: -0.0074414, -0.0028365, -0.0076618, -0.0028725, -0.0032302, 0.0033306
8: -0.0034775, -0.0010558, -0.0035934, -0.0010748, -0.0016987, 0.0017515
9: -0.0006396, 0.0021685, -0.0006176, 0.0023029, -0.0020310, 0.0019698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0015086
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0015185
time: 0.91 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.14 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0014910
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015078
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0014420
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0014481
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015532
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015532, upper bound: 0.0015780
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0015086
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.14
Output dim: 0, lower bound: -0.0015185, upper bound: 0.0015185

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893243, 0.9933682, 0.9893228, 0.9934980, -0.0027569, 0.0026460
1: -0.0039241, -0.0029164, -0.0039244, -0.0028841, -0.0006870, 0.0006593
2: 0.0054015, 0.0107414, 0.0052301, 0.0107435, -0.0034941, 0.0036405
3: -0.0061621, -0.0037316, -0.0061631, -0.0036536, -0.0016570, 0.0015904
4: 0.0015733, 0.0026069, 0.0015402, 0.0026073, -0.0006763, 0.0007046
5: 0.0057531, 0.0124693, 0.0055375, 0.0124719, -0.0043946, 0.0045788
6: -0.0016240, 0.0000806, -0.0016247, 0.0001353, -0.0011621, 0.0011154
7: -0.0073394, -0.0029290, -0.0073412, -0.0027874, -0.0030068, 0.0028859
8: -0.0034239, -0.0011045, -0.0034248, -0.0010300, -0.0015813, 0.0015177
9: -0.0005831, 0.0021063, -0.0006695, 0.0021074, -0.0017598, 0.0018335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0014287
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014287
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890304, 0.9932626, 0.9893228, 0.9934980, -0.0030731, 0.0025736
1: -0.0039973, -0.0029427, -0.0039244, -0.0028841, -0.0007657, 0.0006413
2: 0.0055409, 0.0111295, 0.0052301, 0.0107435, -0.0033985, 0.0040580
3: -0.0063388, -0.0037951, -0.0061631, -0.0036536, -0.0018470, 0.0015468
4: 0.0016003, 0.0026820, 0.0015402, 0.0026073, -0.0006578, 0.0007854
5: 0.0059285, 0.0129575, 0.0055375, 0.0124719, -0.0042744, 0.0051039
6: -0.0017479, 0.0000361, -0.0016247, 0.0001353, -0.0012954, 0.0010849
7: -0.0076600, -0.0030441, -0.0073412, -0.0027874, -0.0033516, 0.0028069
8: -0.0035925, -0.0011650, -0.0034248, -0.0010300, -0.0017626, 0.0014761
9: -0.0005129, 0.0023018, -0.0006695, 0.0021074, -0.0017116, 0.0020438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0014481
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014481
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9893339, 0.9932981, 0.9890301, 0.9933873, -0.0028417, 0.0028672
1: -0.0039217, -0.0029339, -0.0039974, -0.0029117, -0.0007081, 0.0007144
2: 0.0054942, 0.0107288, 0.0053763, 0.0111299, -0.0037861, 0.0037525
3: -0.0061564, -0.0037738, -0.0063390, -0.0037202, -0.0017080, 0.0017233
4: 0.0015913, 0.0026044, 0.0015685, 0.0026821, -0.0007328, 0.0007263
5: 0.0058697, 0.0124534, 0.0057214, 0.0129579, -0.0047619, 0.0047196
6: -0.0016200, 0.0000510, -0.0017480, 0.0000887, -0.0011979, 0.0012086
7: -0.0073290, -0.0030055, -0.0076603, -0.0029082, -0.0030993, 0.0031271
8: -0.0034184, -0.0011447, -0.0035926, -0.0010935, -0.0016299, 0.0016445
9: -0.0005365, 0.0020999, -0.0005959, 0.0023020, -0.0019069, 0.0018899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014420
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014420
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9892461, 0.9932694, 0.9890338, 0.9933518, -0.0029005, 0.0029352
1: -0.0039435, -0.0029410, -0.0039964, -0.0029205, -0.0007227, 0.0007314
2: 0.0055319, 0.0108447, 0.0054232, 0.0111251, -0.0038759, 0.0038301
3: -0.0062092, -0.0037910, -0.0063368, -0.0037415, -0.0017433, 0.0017642
4: 0.0015986, 0.0026269, 0.0015775, 0.0026811, -0.0007502, 0.0007413
5: 0.0059171, 0.0125992, 0.0057804, 0.0129519, -0.0048749, 0.0048173
6: -0.0016570, 0.0000390, -0.0017465, 0.0000737, -0.0012227, 0.0012373
7: -0.0074248, -0.0030367, -0.0076563, -0.0029469, -0.0031634, 0.0032013
8: -0.0034688, -0.0011611, -0.0035905, -0.0011139, -0.0016636, 0.0016835
9: -0.0005175, 0.0021583, -0.0005722, 0.0022995, -0.0019521, 0.0019290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014481
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014481
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9892308, 0.9934441, 0.9893227, 0.9935172, -0.0028507, 0.0027272
1: -0.0039474, -0.0028975, -0.0039245, -0.0028793, -0.0007103, 0.0006796
2: 0.0053012, 0.0108649, 0.0052046, 0.0107436, -0.0036013, 0.0037644
3: -0.0062184, -0.0036860, -0.0061631, -0.0036420, -0.0017134, 0.0016392
4: 0.0015539, 0.0026308, 0.0015352, 0.0026073, -0.0006970, 0.0007286
5: 0.0056270, 0.0126247, 0.0055055, 0.0124721, -0.0045295, 0.0047346
6: -0.0016634, 0.0001126, -0.0016247, 0.0001435, -0.0012017, 0.0011496
7: -0.0074415, -0.0028462, -0.0073412, -0.0027664, -0.0031091, 0.0029744
8: -0.0034775, -0.0010609, -0.0034248, -0.0010190, -0.0016351, 0.0015642
9: -0.0006337, 0.0021685, -0.0006823, 0.0021074, -0.0018138, 0.0018959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0014924
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014924
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9889287, 0.9933528, 0.9893227, 0.9935172, -0.0031836, 0.0026540
1: -0.0040226, -0.0029203, -0.0039245, -0.0028793, -0.0007933, 0.0006613
2: 0.0054219, 0.0112639, 0.0052046, 0.0107436, -0.0035045, 0.0042040
3: -0.0064000, -0.0037409, -0.0061631, -0.0036420, -0.0019135, 0.0015951
4: 0.0015773, 0.0027080, 0.0015352, 0.0026073, -0.0006783, 0.0008137
5: 0.0057788, 0.0131265, 0.0055055, 0.0124721, -0.0044078, 0.0052875
6: -0.0017908, 0.0000741, -0.0016247, 0.0001435, -0.0013420, 0.0011187
7: -0.0077710, -0.0029459, -0.0073412, -0.0027664, -0.0034722, 0.0028945
8: -0.0036508, -0.0011133, -0.0034248, -0.0010190, -0.0018260, 0.0015222
9: -0.0005729, 0.0023695, -0.0006823, 0.0021074, -0.0017651, 0.0021173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0015185
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0015185
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9892403, 0.9933829, 0.9890301, 0.9934111, -0.0029347, 0.0029472
1: -0.0039450, -0.0029128, -0.0039974, -0.0029057, -0.0007312, 0.0007344
2: 0.0053821, 0.0108523, 0.0053449, 0.0111300, -0.0038918, 0.0038752
3: -0.0062126, -0.0037228, -0.0063390, -0.0037059, -0.0017638, 0.0017714
4: 0.0015696, 0.0026283, 0.0015624, 0.0026821, -0.0007532, 0.0007500
5: 0.0057287, 0.0126088, 0.0056819, 0.0129580, -0.0048948, 0.0048740
6: -0.0016594, 0.0000868, -0.0017480, 0.0000987, -0.0012371, 0.0012424
7: -0.0074311, -0.0029130, -0.0076604, -0.0028823, -0.0032007, 0.0032144
8: -0.0034721, -0.0010961, -0.0035926, -0.0010799, -0.0016832, 0.0016904
9: -0.0005929, 0.0021622, -0.0006117, 0.0023020, -0.0019601, 0.0019518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015085, upper bound: 0.0015086
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015085, upper bound: 0.0015086
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9891502, 0.9933589, 0.9890338, 0.9933755, -0.0029915, 0.0030126
1: -0.0039675, -0.0029187, -0.0039964, -0.0029146, -0.0007454, 0.0007507
2: 0.0054138, 0.0109715, 0.0053918, 0.0111251, -0.0039782, 0.0039503
3: -0.0062669, -0.0037372, -0.0063368, -0.0037272, -0.0017980, 0.0018107
4: 0.0015757, 0.0026514, 0.0015715, 0.0026811, -0.0007700, 0.0007646
5: 0.0057685, 0.0127587, 0.0057409, 0.0129519, -0.0050035, 0.0049684
6: -0.0016975, 0.0000767, -0.0017465, 0.0000837, -0.0012610, 0.0012699
7: -0.0075295, -0.0029391, -0.0076563, -0.0029210, -0.0032627, 0.0032857
8: -0.0035238, -0.0011098, -0.0035905, -0.0011003, -0.0017158, 0.0017279
9: -0.0005770, 0.0022222, -0.0005880, 0.0022996, -0.0020036, 0.0019896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015185
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015185
time: 1.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0014287
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014287
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0014481
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014481
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014420
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014420
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014481
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014743, upper bound: 0.0014481
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0014924
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0014924
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014929, upper bound: 0.0015185
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014893, upper bound: 0.0015185
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0015085, upper bound: 0.0015086
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0015085, upper bound: 0.0015086
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015185
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -0.0014481, upper bound: 0.0015185

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9893254, 0.9933602, 0.9893322, 0.9934316, -0.0026666, 0.0026197
1: -0.0039238, -0.0029184, -0.0039221, -0.0029006, -0.0006644, 0.0006528
2: 0.0054121, 0.0107399, 0.0053178, 0.0107310, -0.0034593, 0.0035212
3: -0.0061615, -0.0037365, -0.0061574, -0.0036936, -0.0016027, 0.0015745
4: 0.0015754, 0.0026066, 0.0015571, 0.0026049, -0.0006695, 0.0006815
5: 0.0057664, 0.0124675, 0.0056479, 0.0124563, -0.0043509, 0.0044287
6: -0.0016235, 0.0000772, -0.0016207, 0.0001073, -0.0011241, 0.0011043
7: -0.0073382, -0.0029378, -0.0073309, -0.0028599, -0.0029083, 0.0028572
8: -0.0034233, -0.0011091, -0.0034194, -0.0010681, -0.0015294, 0.0015026
9: -0.0005778, 0.0021056, -0.0006253, 0.0021011, -0.0017423, 0.0017735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0014382
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0014387
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9893290, 0.9933133, 0.9892452, 0.9933913, -0.0027464, 0.0026867
1: -0.0039229, -0.0029301, -0.0039438, -0.0029107, -0.0006843, 0.0006694
2: 0.0054741, 0.0107352, 0.0053710, 0.0108459, -0.0035477, 0.0036265
3: -0.0061593, -0.0037647, -0.0062097, -0.0037178, -0.0016506, 0.0016148
4: 0.0015874, 0.0026057, 0.0015674, 0.0026271, -0.0006867, 0.0007019
5: 0.0058445, 0.0124615, 0.0057148, 0.0126007, -0.0044621, 0.0045612
6: -0.0016220, 0.0000574, -0.0016574, 0.0000904, -0.0011577, 0.0011325
7: -0.0073343, -0.0029890, -0.0074257, -0.0029038, -0.0029953, 0.0029302
8: -0.0034212, -0.0011360, -0.0034693, -0.0010912, -0.0015752, 0.0015410
9: -0.0005466, 0.0021032, -0.0005985, 0.0021589, -0.0017868, 0.0018265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014015, upper bound: 0.0014138
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014765, upper bound: 0.0014137
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9890317, 0.9932536, 0.9893322, 0.9934316, -0.0029831, 0.0025437
1: -0.0039970, -0.0029450, -0.0039221, -0.0029006, -0.0007433, 0.0006338
2: 0.0055529, 0.0111278, 0.0053178, 0.0107310, -0.0033589, 0.0039391
3: -0.0063380, -0.0038005, -0.0061574, -0.0036936, -0.0017929, 0.0015288
4: 0.0016026, 0.0026817, 0.0015571, 0.0026049, -0.0006501, 0.0007624
5: 0.0059435, 0.0129553, 0.0056479, 0.0124563, -0.0042246, 0.0049544
6: -0.0017474, 0.0000323, -0.0016207, 0.0001073, -0.0012575, 0.0010723
7: -0.0076586, -0.0030540, -0.0073309, -0.0028599, -0.0032535, 0.0027743
8: -0.0035917, -0.0011702, -0.0034194, -0.0010681, -0.0017110, 0.0014590
9: -0.0005069, 0.0023009, -0.0006253, 0.0021011, -0.0016917, 0.0019839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014420
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014481
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9890353, 0.9932176, 0.9892452, 0.9933913, -0.0030615, 0.0026349
1: -0.0039961, -0.0029539, -0.0039438, -0.0029107, -0.0007628, 0.0006565
2: 0.0056003, 0.0111231, 0.0053710, 0.0108459, -0.0034793, 0.0040427
3: -0.0063359, -0.0038221, -0.0062097, -0.0037178, -0.0018401, 0.0015836
4: 0.0016118, 0.0026807, 0.0015674, 0.0026271, -0.0006734, 0.0007825
5: 0.0060031, 0.0129494, 0.0057148, 0.0126007, -0.0043761, 0.0050846
6: -0.0017459, 0.0000172, -0.0016574, 0.0000904, -0.0012905, 0.0011107
7: -0.0076547, -0.0030932, -0.0074257, -0.0029038, -0.0033390, 0.0028737
8: -0.0035897, -0.0011908, -0.0034693, -0.0010912, -0.0017560, 0.0015112
9: -0.0004830, 0.0022985, -0.0005985, 0.0021589, -0.0017524, 0.0020361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013909, upper bound: 0.0014218
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014218
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9893339, 0.9932981, 0.9890317, 0.9932536, -0.0027066, 0.0028459
1: -0.0039217, -0.0029339, -0.0039970, -0.0029450, -0.0006744, 0.0007091
2: 0.0054942, 0.0107288, 0.0055529, 0.0111278, -0.0037580, 0.0035740
3: -0.0061564, -0.0037738, -0.0063380, -0.0038005, -0.0016267, 0.0017105
4: 0.0015913, 0.0026044, 0.0016026, 0.0026817, -0.0007274, 0.0006917
5: 0.0058697, 0.0124534, 0.0059435, 0.0129553, -0.0047266, 0.0044952
6: -0.0016200, 0.0000510, -0.0017474, 0.0000323, -0.0011409, 0.0011997
7: -0.0073290, -0.0030055, -0.0076586, -0.0030540, -0.0029519, 0.0031039
8: -0.0034184, -0.0011447, -0.0035917, -0.0011702, -0.0015524, 0.0016323
9: -0.0005365, 0.0020999, -0.0005069, 0.0023009, -0.0018927, 0.0018001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014303
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014420
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9893339, 0.9932981, 0.9889299, 0.9933437, -0.0027726, 0.0029351
1: -0.0039217, -0.0029339, -0.0040223, -0.0029225, -0.0006909, 0.0007313
2: 0.0054942, 0.0107288, 0.0054338, 0.0112622, -0.0038758, 0.0036612
3: -0.0061564, -0.0037738, -0.0063992, -0.0037463, -0.0016664, 0.0017641
4: 0.0015913, 0.0026044, 0.0015796, 0.0027077, -0.0007501, 0.0007086
5: 0.0058697, 0.0124534, 0.0057937, 0.0131243, -0.0048747, 0.0046048
6: -0.0016200, 0.0000510, -0.0017903, 0.0000703, -0.0011688, 0.0012372
7: -0.0073290, -0.0030055, -0.0077696, -0.0029557, -0.0030239, 0.0032011
8: -0.0034184, -0.0011447, -0.0036501, -0.0011185, -0.0015903, 0.0016834
9: -0.0005365, 0.0020999, -0.0005669, 0.0023686, -0.0019520, 0.0018440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014303
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014420
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9892461, 0.9932694, 0.9890353, 0.9932176, -0.0027656, 0.0029139
1: -0.0039435, -0.0029410, -0.0039961, -0.0029539, -0.0006891, 0.0007261
2: 0.0055319, 0.0108447, 0.0056003, 0.0111231, -0.0038477, 0.0036520
3: -0.0062092, -0.0037910, -0.0063359, -0.0038221, -0.0016622, 0.0017513
4: 0.0015986, 0.0026269, 0.0016118, 0.0026807, -0.0007447, 0.0007068
5: 0.0059171, 0.0125992, 0.0060031, 0.0129494, -0.0048394, 0.0045932
6: -0.0016570, 0.0000390, -0.0017459, 0.0000172, -0.0011658, 0.0012283
7: -0.0074248, -0.0030367, -0.0076547, -0.0030932, -0.0030163, 0.0031780
8: -0.0034688, -0.0011611, -0.0035897, -0.0011908, -0.0015863, 0.0016713
9: -0.0005175, 0.0021583, -0.0004830, 0.0022985, -0.0019379, 0.0018393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014272
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014481
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9892461, 0.9932694, 0.9889337, 0.9933070, -0.0028319, 0.0030025
1: -0.0039435, -0.0029410, -0.0040214, -0.0029317, -0.0007056, 0.0007481
2: 0.0055319, 0.0108447, 0.0054823, 0.0112571, -0.0039648, 0.0037395
3: -0.0062092, -0.0037910, -0.0063969, -0.0037684, -0.0017021, 0.0018046
4: 0.0015986, 0.0026269, 0.0015890, 0.0027067, -0.0007674, 0.0007238
5: 0.0059171, 0.0125992, 0.0058548, 0.0131180, -0.0049867, 0.0047033
6: -0.0016570, 0.0000390, -0.0017887, 0.0000548, -0.0011938, 0.0012657
7: -0.0074248, -0.0030367, -0.0077654, -0.0029958, -0.0030886, 0.0032747
8: -0.0034688, -0.0011611, -0.0036479, -0.0011396, -0.0016243, 0.0017221
9: -0.0005175, 0.0021583, -0.0005424, 0.0023661, -0.0019969, 0.0018834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014272
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014481
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9892319, 0.9934360, 0.9893321, 0.9934523, -0.0027617, 0.0027009
1: -0.0039471, -0.0028995, -0.0039221, -0.0028955, -0.0006881, 0.0006730
2: 0.0053118, 0.0108635, 0.0052905, 0.0107311, -0.0035666, 0.0036467
3: -0.0062177, -0.0036908, -0.0061574, -0.0036811, -0.0016598, 0.0016233
4: 0.0015560, 0.0026305, 0.0015519, 0.0026049, -0.0006903, 0.0007058
5: 0.0056404, 0.0126229, 0.0056136, 0.0124563, -0.0044858, 0.0045866
6: -0.0016630, 0.0001092, -0.0016207, 0.0001160, -0.0011641, 0.0011385
7: -0.0074403, -0.0028550, -0.0073309, -0.0028374, -0.0030120, 0.0029458
8: -0.0034769, -0.0010655, -0.0034194, -0.0010563, -0.0015840, 0.0015491
9: -0.0006283, 0.0021678, -0.0006390, 0.0021011, -0.0017963, 0.0018367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0015004
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0015014
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9892358, 0.9933904, 0.9892453, 0.9934124, -0.0028423, 0.0027714
1: -0.0039461, -0.0029109, -0.0039438, -0.0029054, -0.0007082, 0.0006905
2: 0.0053722, 0.0108583, 0.0053431, 0.0108458, -0.0036595, 0.0037532
3: -0.0062153, -0.0037183, -0.0062097, -0.0037050, -0.0017083, 0.0016657
4: 0.0015677, 0.0026295, 0.0015620, 0.0026271, -0.0007083, 0.0007264
5: 0.0057162, 0.0126163, 0.0056796, 0.0126006, -0.0046027, 0.0047206
6: -0.0016613, 0.0000900, -0.0016573, 0.0000993, -0.0011981, 0.0011682
7: -0.0074360, -0.0029048, -0.0074257, -0.0028807, -0.0030999, 0.0030225
8: -0.0034747, -0.0010917, -0.0034692, -0.0010791, -0.0016302, 0.0015895
9: -0.0005979, 0.0021652, -0.0006126, 0.0021589, -0.0018431, 0.0018903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014015, upper bound: 0.0014766
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014765, upper bound: 0.0014766
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9889299, 0.9933437, 0.9893321, 0.9934523, -0.0030949, 0.0026239
1: -0.0040223, -0.0029225, -0.0039221, -0.0028955, -0.0007712, 0.0006538
2: 0.0054338, 0.0112622, 0.0052905, 0.0107311, -0.0034649, 0.0040868
3: -0.0063992, -0.0037463, -0.0061574, -0.0036811, -0.0018601, 0.0015771
4: 0.0015796, 0.0027077, 0.0015519, 0.0026049, -0.0006706, 0.0007910
5: 0.0057937, 0.0131243, 0.0056136, 0.0124563, -0.0043579, 0.0051401
6: -0.0017903, 0.0000703, -0.0016207, 0.0001160, -0.0013046, 0.0011061
7: -0.0077696, -0.0029557, -0.0073309, -0.0028374, -0.0033754, 0.0028618
8: -0.0036501, -0.0011185, -0.0034194, -0.0010563, -0.0017751, 0.0015050
9: -0.0005669, 0.0023686, -0.0006390, 0.0021011, -0.0017451, 0.0020583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015086
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015185
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9889337, 0.9933070, 0.9892453, 0.9934124, -0.0031747, 0.0027178
1: -0.0040214, -0.0029317, -0.0039438, -0.0029054, -0.0007910, 0.0006772
2: 0.0054823, 0.0112571, 0.0053431, 0.0108458, -0.0035888, 0.0041921
3: -0.0063969, -0.0037684, -0.0062097, -0.0037050, -0.0019081, 0.0016335
4: 0.0015890, 0.0027067, 0.0015620, 0.0026271, -0.0006946, 0.0008114
5: 0.0058548, 0.0131180, 0.0056796, 0.0126006, -0.0045138, 0.0052726
6: -0.0017887, 0.0000548, -0.0016573, 0.0000993, -0.0013382, 0.0011457
7: -0.0077654, -0.0029958, -0.0074257, -0.0028807, -0.0034624, 0.0029642
8: -0.0036479, -0.0011396, -0.0034692, -0.0010791, -0.0018208, 0.0015588
9: -0.0005424, 0.0023661, -0.0006126, 0.0021589, -0.0018075, 0.0021114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013909, upper bound: 0.0014911
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014911
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9892403, 0.9933829, 0.9890400, 0.9933451, -0.0028615, 0.0029360
1: -0.0039450, -0.0029128, -0.0039949, -0.0029222, -0.0007130, 0.0007316
2: 0.0053821, 0.0108523, 0.0054320, 0.0111169, -0.0038769, 0.0037785
3: -0.0062126, -0.0037228, -0.0063330, -0.0037455, -0.0017198, 0.0017646
4: 0.0015696, 0.0026283, 0.0015792, 0.0026795, -0.0007504, 0.0007313
5: 0.0057287, 0.0126088, 0.0057914, 0.0129416, -0.0048761, 0.0047524
6: -0.0016594, 0.0000868, -0.0017439, 0.0000709, -0.0012062, 0.0012376
7: -0.0074311, -0.0029130, -0.0076496, -0.0029542, -0.0031208, 0.0032021
8: -0.0034721, -0.0010961, -0.0035870, -0.0011177, -0.0016412, 0.0016839
9: -0.0005929, 0.0021622, -0.0005678, 0.0022954, -0.0019526, 0.0019031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014919
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015086
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9892403, 0.9933829, 0.9889082, 0.9933501, -0.0028250, 0.0031477
1: -0.0039450, -0.0029128, -0.0040277, -0.0029209, -0.0007039, 0.0007843
2: 0.0053821, 0.0108523, 0.0054254, 0.0112909, -0.0041565, 0.0037304
3: -0.0062126, -0.0037228, -0.0064122, -0.0037425, -0.0016979, 0.0018919
4: 0.0015696, 0.0026283, 0.0015780, 0.0027132, -0.0008045, 0.0007220
5: 0.0057287, 0.0126088, 0.0057832, 0.0131604, -0.0052278, 0.0046919
6: -0.0016594, 0.0000868, -0.0017994, 0.0000730, -0.0011909, 0.0013269
7: -0.0074311, -0.0029130, -0.0077933, -0.0029488, -0.0030811, 0.0034330
8: -0.0034721, -0.0010961, -0.0036625, -0.0011149, -0.0016203, 0.0018054
9: -0.0005929, 0.0021622, -0.0005711, 0.0023831, -0.0020935, 0.0018788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014919
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015086
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9891502, 0.9933589, 0.9890353, 0.9932176, -0.0028350, 0.0029892
1: -0.0039675, -0.0029187, -0.0039961, -0.0029539, -0.0007064, 0.0007448
2: 0.0054138, 0.0109715, 0.0056003, 0.0111231, -0.0039471, 0.0037436
3: -0.0062669, -0.0037372, -0.0063359, -0.0038221, -0.0017039, 0.0017966
4: 0.0015757, 0.0026514, 0.0016118, 0.0026807, -0.0007640, 0.0007246
5: 0.0057685, 0.0127587, 0.0060031, 0.0129494, -0.0049645, 0.0047085
6: -0.0016975, 0.0000767, -0.0017459, 0.0000172, -0.0011951, 0.0012600
7: -0.0075295, -0.0029391, -0.0076547, -0.0030932, -0.0030920, 0.0032601
8: -0.0035238, -0.0011098, -0.0035897, -0.0011908, -0.0016260, 0.0017145
9: -0.0005770, 0.0022222, -0.0004830, 0.0022985, -0.0019880, 0.0018855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014893
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015185
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9891502, 0.9933589, 0.9889337, 0.9933070, -0.0028552, 0.0029944
1: -0.0039675, -0.0029187, -0.0040214, -0.0029317, -0.0007114, 0.0007461
2: 0.0054138, 0.0109715, 0.0054823, 0.0112571, -0.0039540, 0.0037702
3: -0.0062669, -0.0037372, -0.0063969, -0.0037684, -0.0017160, 0.0017997
4: 0.0015757, 0.0026514, 0.0015890, 0.0027067, -0.0007653, 0.0007297
5: 0.0057685, 0.0127587, 0.0058548, 0.0131180, -0.0049731, 0.0047419
6: -0.0016975, 0.0000767, -0.0017887, 0.0000548, -0.0012036, 0.0012622
7: -0.0075295, -0.0029391, -0.0077654, -0.0029958, -0.0031140, 0.0032658
8: -0.0035238, -0.0011098, -0.0036479, -0.0011396, -0.0016376, 0.0017174
9: -0.0005770, 0.0022222, -0.0005424, 0.0023661, -0.0019915, 0.0018989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014893
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015185
time: 1.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0014382
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0014387
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014015, upper bound: 0.0014138
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014765, upper bound: 0.0014137
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014420
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014481
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0013909, upper bound: 0.0014218
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014218
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014303
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014420
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014303
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014420
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014272
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014481
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014272
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014493, upper bound: 0.0014481
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0015004
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0015004, upper bound: 0.0015014
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014015, upper bound: 0.0014766
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014765, upper bound: 0.0014766
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015086
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015185
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0013909, upper bound: 0.0014911
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014911
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014919
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015086
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014919
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0015086
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014893
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015185
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014893
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015185

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9893322, 0.9934316, -0.0026530, 0.0025418
1: -0.0039217, -0.0029332, -0.0039221, -0.0029006, -0.0006611, 0.0006333
2: 0.0054905, 0.0107292, 0.0053178, 0.0107310, -0.0033564, 0.0035032
3: -0.0061566, -0.0037722, -0.0061574, -0.0036936, -0.0015945, 0.0015277
4: 0.0015906, 0.0026045, 0.0015571, 0.0026049, -0.0006496, 0.0006780
5: 0.0058651, 0.0124540, 0.0056479, 0.0124563, -0.0042215, 0.0044061
6: -0.0016201, 0.0000522, -0.0016207, 0.0001073, -0.0011183, 0.0010715
7: -0.0073294, -0.0030025, -0.0073309, -0.0028599, -0.0028935, 0.0027722
8: -0.0034186, -0.0011432, -0.0034194, -0.0010681, -0.0015216, 0.0014579
9: -0.0005383, 0.0021002, -0.0006253, 0.0021011, -0.0016905, 0.0017644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014382
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014383
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9892465, 0.9932592, 0.9893322, 0.9934316, -0.0027704, 0.0025046
1: -0.0039434, -0.0029436, -0.0039221, -0.0029006, -0.0006903, 0.0006241
2: 0.0055454, 0.0108441, 0.0053178, 0.0107310, -0.0033073, 0.0036582
3: -0.0062089, -0.0037972, -0.0061574, -0.0036936, -0.0016651, 0.0015054
4: 0.0016012, 0.0026267, 0.0015571, 0.0026049, -0.0006401, 0.0007080
5: 0.0059341, 0.0125985, 0.0056479, 0.0124563, -0.0041598, 0.0046011
6: -0.0016568, 0.0000347, -0.0016207, 0.0001073, -0.0011678, 0.0010558
7: -0.0074242, -0.0030479, -0.0073309, -0.0028599, -0.0030215, 0.0027317
8: -0.0034685, -0.0011670, -0.0034194, -0.0010681, -0.0015890, 0.0014365
9: -0.0005107, 0.0021580, -0.0006253, 0.0021011, -0.0016657, 0.0018425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014387
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014387
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894422, 0.9932245, 0.9892803, 0.9933891, -0.0026423, 0.0026274
1: -0.0038947, -0.0029522, -0.0039350, -0.0029112, -0.0006584, 0.0006547
2: 0.0055913, 0.0105857, 0.0053739, 0.0107995, -0.0034694, 0.0034891
3: -0.0060913, -0.0038180, -0.0061886, -0.0037191, -0.0015881, 0.0015791
4: 0.0016101, 0.0025767, 0.0015680, 0.0026181, -0.0006715, 0.0006753
5: 0.0059918, 0.0122735, 0.0057184, 0.0125424, -0.0043636, 0.0043884
6: -0.0015743, 0.0000200, -0.0016426, 0.0000894, -0.0011138, 0.0011075
7: -0.0072108, -0.0030858, -0.0073874, -0.0029062, -0.0028818, 0.0028655
8: -0.0033563, -0.0011869, -0.0034491, -0.0010925, -0.0015155, 0.0015069
9: -0.0004876, 0.0020279, -0.0005971, 0.0021356, -0.0017474, 0.0017573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013621, upper bound: 0.0014138
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013621, upper bound: 0.0014137
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9893777, 0.9933112, 0.9892604, 0.9933904, -0.0026770, 0.0026552
1: -0.0039107, -0.0029306, -0.0039400, -0.0029109, -0.0006670, 0.0006616
2: 0.0054768, 0.0106709, 0.0053722, 0.0108259, -0.0035062, 0.0035349
3: -0.0061301, -0.0037659, -0.0062006, -0.0037183, -0.0016090, 0.0015959
4: 0.0015879, 0.0025932, 0.0015677, 0.0026232, -0.0006786, 0.0006842
5: 0.0058478, 0.0123807, 0.0057163, 0.0125756, -0.0044099, 0.0044460
6: -0.0016015, 0.0000566, -0.0016510, 0.0000900, -0.0011284, 0.0011193
7: -0.0072812, -0.0029912, -0.0074093, -0.0029048, -0.0029196, 0.0028959
8: -0.0033933, -0.0011372, -0.0034606, -0.0010918, -0.0015354, 0.0015229
9: -0.0005452, 0.0020708, -0.0005979, 0.0021489, -0.0017659, 0.0017804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014358, upper bound: 0.0014138
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014358, upper bound: 0.0014138
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9893322, 0.9934316, -0.0029718, 0.0024421
1: -0.0039946, -0.0029617, -0.0039221, -0.0029006, -0.0007405, 0.0006085
2: 0.0056415, 0.0111150, 0.0053178, 0.0107310, -0.0032248, 0.0039242
3: -0.0063322, -0.0038409, -0.0061574, -0.0036936, -0.0017861, 0.0014678
4: 0.0016198, 0.0026792, 0.0015571, 0.0026049, -0.0006241, 0.0007595
5: 0.0060549, 0.0129393, 0.0056479, 0.0124563, -0.0040559, 0.0049357
6: -0.0017433, 0.0000040, -0.0016207, 0.0001073, -0.0012527, 0.0010294
7: -0.0076480, -0.0031272, -0.0073309, -0.0028599, -0.0032412, 0.0026635
8: -0.0035862, -0.0012087, -0.0034194, -0.0010681, -0.0017045, 0.0014007
9: -0.0004623, 0.0022945, -0.0006253, 0.0021011, -0.0016242, 0.0019765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014422
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014423
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9889095, 0.9931933, 0.9893322, 0.9934316, -0.0031809, 0.0024762
1: -0.0040274, -0.0029600, -0.0039221, -0.0029006, -0.0007926, 0.0006170
2: 0.0056324, 0.0112891, 0.0053178, 0.0107310, -0.0032698, 0.0042004
3: -0.0064114, -0.0038367, -0.0061574, -0.0036936, -0.0019118, 0.0014883
4: 0.0016180, 0.0027129, 0.0015571, 0.0026049, -0.0006329, 0.0008130
5: 0.0060435, 0.0131582, 0.0056479, 0.0124563, -0.0041125, 0.0052829
6: -0.0017989, 0.0000069, -0.0016207, 0.0001073, -0.0013409, 0.0010438
7: -0.0077918, -0.0031197, -0.0073309, -0.0028599, -0.0034692, 0.0027006
8: -0.0036618, -0.0012048, -0.0034194, -0.0010681, -0.0018244, 0.0014202
9: -0.0004669, 0.0023822, -0.0006253, 0.0021011, -0.0016468, 0.0021155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014481
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014481
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891484, 0.9931169, 0.9892803, 0.9933891, -0.0029402, 0.0025546
1: -0.0039679, -0.0029790, -0.0039350, -0.0029112, -0.0007326, 0.0006365
2: 0.0057333, 0.0109737, 0.0053739, 0.0107995, -0.0033734, 0.0038825
3: -0.0062679, -0.0038827, -0.0061886, -0.0037191, -0.0017672, 0.0015354
4: 0.0016376, 0.0026518, 0.0015680, 0.0026181, -0.0006529, 0.0007515
5: 0.0061705, 0.0127615, 0.0057184, 0.0125424, -0.0042428, 0.0048832
6: -0.0016982, -0.0000253, -0.0016426, 0.0000894, -0.0012394, 0.0010769
7: -0.0075313, -0.0032031, -0.0073874, -0.0029062, -0.0032067, 0.0027862
8: -0.0035248, -0.0012486, -0.0034491, -0.0010925, -0.0016864, 0.0014652
9: -0.0004160, 0.0022233, -0.0005971, 0.0021356, -0.0016990, 0.0019554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013521, upper bound: 0.0014218
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013521, upper bound: 0.0014218
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9890829, 0.9932152, 0.9892604, 0.9933904, -0.0029819, 0.0026050
1: -0.0039842, -0.0029546, -0.0039400, -0.0029109, -0.0007430, 0.0006491
2: 0.0056036, 0.0110602, 0.0053722, 0.0108259, -0.0034398, 0.0039375
3: -0.0063072, -0.0038236, -0.0062006, -0.0037183, -0.0017922, 0.0015657
4: 0.0016124, 0.0026686, 0.0015677, 0.0026232, -0.0006658, 0.0007621
5: 0.0060073, 0.0128703, 0.0057163, 0.0125756, -0.0043264, 0.0049524
6: -0.0017258, 0.0000161, -0.0016510, 0.0000900, -0.0012570, 0.0010981
7: -0.0076027, -0.0030959, -0.0074093, -0.0029048, -0.0032521, 0.0028411
8: -0.0035624, -0.0011923, -0.0034606, -0.0010918, -0.0017103, 0.0014941
9: -0.0004814, 0.0022669, -0.0005979, 0.0021489, -0.0017325, 0.0019831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014218
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014218
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9890317, 0.9932536, -0.0025217, 0.0028499
1: -0.0039217, -0.0029332, -0.0039970, -0.0029450, -0.0006283, 0.0007101
2: 0.0054905, 0.0107292, 0.0055529, 0.0111278, -0.0037633, 0.0033299
3: -0.0061566, -0.0037722, -0.0063380, -0.0038005, -0.0015156, 0.0017129
4: 0.0015906, 0.0026045, 0.0016026, 0.0026817, -0.0007284, 0.0006445
5: 0.0058651, 0.0124540, 0.0059435, 0.0129553, -0.0047332, 0.0041881
6: -0.0016201, 0.0000522, -0.0017474, 0.0000323, -0.0010630, 0.0012013
7: -0.0073294, -0.0030025, -0.0076586, -0.0030540, -0.0027503, 0.0031082
8: -0.0034186, -0.0011432, -0.0035917, -0.0011702, -0.0014463, 0.0016346
9: -0.0005383, 0.0021002, -0.0005069, 0.0023009, -0.0018954, 0.0016771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014519
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014519
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9890317, 0.9932536, -0.0027004, 0.0026381
1: -0.0039946, -0.0029617, -0.0039970, -0.0029450, -0.0006729, 0.0006573
2: 0.0056415, 0.0111150, 0.0055529, 0.0111278, -0.0034836, 0.0035659
3: -0.0063322, -0.0038409, -0.0063380, -0.0038005, -0.0016230, 0.0015856
4: 0.0016198, 0.0026792, 0.0016026, 0.0026817, -0.0006742, 0.0006902
5: 0.0060549, 0.0129393, 0.0059435, 0.0129553, -0.0043814, 0.0044849
6: -0.0017433, 0.0000040, -0.0017474, 0.0000323, -0.0011383, 0.0011120
7: -0.0076480, -0.0031272, -0.0076586, -0.0030540, -0.0029452, 0.0028772
8: -0.0035862, -0.0012087, -0.0035917, -0.0011702, -0.0015488, 0.0015131
9: -0.0004623, 0.0022945, -0.0005069, 0.0023009, -0.0017545, 0.0017960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014653
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014654
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9889299, 0.9933437, -0.0025898, 0.0029391
1: -0.0039217, -0.0029332, -0.0040223, -0.0029225, -0.0006453, 0.0007323
2: 0.0054905, 0.0107292, 0.0054338, 0.0112622, -0.0038810, 0.0034197
3: -0.0061566, -0.0037722, -0.0063992, -0.0037463, -0.0015565, 0.0017665
4: 0.0015906, 0.0026045, 0.0015796, 0.0027077, -0.0007512, 0.0006619
5: 0.0058651, 0.0124540, 0.0057937, 0.0131243, -0.0048813, 0.0043011
6: -0.0016201, 0.0000522, -0.0017903, 0.0000703, -0.0010917, 0.0012389
7: -0.0073294, -0.0030025, -0.0077696, -0.0029557, -0.0028245, 0.0032055
8: -0.0034186, -0.0011432, -0.0036501, -0.0011185, -0.0014854, 0.0016857
9: -0.0005383, 0.0021002, -0.0005669, 0.0023686, -0.0019547, 0.0017224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014303
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014303
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9889299, 0.9933437, -0.0027665, 0.0027090
1: -0.0039946, -0.0029617, -0.0040223, -0.0029225, -0.0006893, 0.0006750
2: 0.0056415, 0.0111150, 0.0054338, 0.0112622, -0.0035771, 0.0036531
3: -0.0063322, -0.0038409, -0.0063992, -0.0037463, -0.0016627, 0.0016282
4: 0.0016198, 0.0026792, 0.0015796, 0.0027077, -0.0006923, 0.0007070
5: 0.0060549, 0.0129393, 0.0057937, 0.0131243, -0.0044991, 0.0045946
6: -0.0017433, 0.0000040, -0.0017903, 0.0000703, -0.0011662, 0.0011419
7: -0.0076480, -0.0031272, -0.0077696, -0.0029557, -0.0030172, 0.0029545
8: -0.0035862, -0.0012087, -0.0036501, -0.0011185, -0.0015867, 0.0015537
9: -0.0004623, 0.0022945, -0.0005669, 0.0023686, -0.0018016, 0.0018399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014420
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014420
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9892465, 0.9932592, 0.9890353, 0.9932176, -0.0026121, 0.0029252
1: -0.0039434, -0.0029436, -0.0039961, -0.0029539, -0.0006509, 0.0007289
2: 0.0055454, 0.0108441, 0.0056003, 0.0111231, -0.0038627, 0.0034493
3: -0.0062089, -0.0037972, -0.0063359, -0.0038221, -0.0015700, 0.0017581
4: 0.0016012, 0.0026267, 0.0016118, 0.0026807, -0.0007476, 0.0006676
5: 0.0059341, 0.0125985, 0.0060031, 0.0129494, -0.0048582, 0.0043383
6: -0.0016568, 0.0000347, -0.0017459, 0.0000172, -0.0011011, 0.0012331
7: -0.0074242, -0.0030479, -0.0076547, -0.0030932, -0.0028489, 0.0031903
8: -0.0034685, -0.0011670, -0.0035897, -0.0011908, -0.0014982, 0.0016778
9: -0.0005107, 0.0021580, -0.0004830, 0.0022985, -0.0019455, 0.0017372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0013514
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014236
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9889095, 0.9931933, 0.9890353, 0.9932176, -0.0027606, 0.0027567
1: -0.0040274, -0.0029600, -0.0039961, -0.0029539, -0.0006879, 0.0006869
2: 0.0056324, 0.0112891, 0.0056003, 0.0111231, -0.0036402, 0.0036454
3: -0.0064114, -0.0038367, -0.0063359, -0.0038221, -0.0016592, 0.0016569
4: 0.0016180, 0.0027129, 0.0016118, 0.0026807, -0.0007046, 0.0007056
5: 0.0060435, 0.0131582, 0.0060031, 0.0129494, -0.0045784, 0.0045849
6: -0.0017989, 0.0000069, -0.0017459, 0.0000172, -0.0011637, 0.0011621
7: -0.0077918, -0.0031197, -0.0076547, -0.0030932, -0.0030108, 0.0030066
8: -0.0036618, -0.0012048, -0.0035897, -0.0011908, -0.0015834, 0.0015811
9: -0.0004669, 0.0023822, -0.0004830, 0.0022985, -0.0018334, 0.0018360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0013682
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014493
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9892465, 0.9932592, 0.9889337, 0.9933070, -0.0026756, 0.0030138
1: -0.0039434, -0.0029436, -0.0040214, -0.0029317, -0.0006667, 0.0007510
2: 0.0055454, 0.0108441, 0.0054823, 0.0112571, -0.0039797, 0.0035331
3: -0.0062089, -0.0037972, -0.0063969, -0.0037684, -0.0016081, 0.0018114
4: 0.0016012, 0.0026267, 0.0015890, 0.0027067, -0.0007703, 0.0006838
5: 0.0059341, 0.0125985, 0.0058548, 0.0131180, -0.0050055, 0.0044437
6: -0.0016568, 0.0000347, -0.0017887, 0.0000548, -0.0011278, 0.0012704
7: -0.0074242, -0.0030479, -0.0077654, -0.0029958, -0.0029181, 0.0032870
8: -0.0034685, -0.0011670, -0.0036479, -0.0011396, -0.0015346, 0.0017286
9: -0.0005107, 0.0021580, -0.0005424, 0.0023661, -0.0020044, 0.0017794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0013378
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014011
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9889095, 0.9931933, 0.9889337, 0.9933070, -0.0028269, 0.0028267
1: -0.0040274, -0.0029600, -0.0040214, -0.0029317, -0.0007044, 0.0007043
2: 0.0056324, 0.0112891, 0.0054823, 0.0112571, -0.0037327, 0.0037329
3: -0.0064114, -0.0038367, -0.0063969, -0.0037684, -0.0016991, 0.0016990
4: 0.0016180, 0.0027129, 0.0015890, 0.0027067, -0.0007225, 0.0007225
5: 0.0060435, 0.0131582, 0.0058548, 0.0131180, -0.0046947, 0.0046950
6: -0.0017989, 0.0000069, -0.0017887, 0.0000548, -0.0011916, 0.0011916
7: -0.0077918, -0.0031197, -0.0077654, -0.0029958, -0.0030831, 0.0030830
8: -0.0036618, -0.0012048, -0.0036479, -0.0011396, -0.0016214, 0.0016213
9: -0.0004669, 0.0023822, -0.0005424, 0.0023661, -0.0018800, 0.0018801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0013511
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014218
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9893321, 0.9934523, -0.0027471, 0.0026222
1: -0.0039451, -0.0029142, -0.0039221, -0.0028955, -0.0006845, 0.0006534
2: 0.0053900, 0.0108528, 0.0052905, 0.0107311, -0.0034626, 0.0036275
3: -0.0062129, -0.0037264, -0.0061574, -0.0036811, -0.0016511, 0.0015760
4: 0.0015711, 0.0026284, 0.0015519, 0.0026049, -0.0006702, 0.0007021
5: 0.0057386, 0.0126095, 0.0056136, 0.0124563, -0.0043551, 0.0045624
6: -0.0016596, 0.0000843, -0.0016207, 0.0001160, -0.0011580, 0.0011054
7: -0.0074315, -0.0029195, -0.0073309, -0.0028374, -0.0029961, 0.0028599
8: -0.0034723, -0.0010995, -0.0034194, -0.0010563, -0.0015756, 0.0015040
9: -0.0005890, 0.0021624, -0.0006390, 0.0021011, -0.0017440, 0.0018270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015004
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015004
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9891506, 0.9933396, 0.9893321, 0.9934523, -0.0028632, 0.0025868
1: -0.0039673, -0.0029236, -0.0039221, -0.0028955, -0.0007134, 0.0006446
2: 0.0054393, 0.0109708, 0.0052905, 0.0107311, -0.0034159, 0.0037808
3: -0.0062665, -0.0037489, -0.0061574, -0.0036811, -0.0017208, 0.0015548
4: 0.0015807, 0.0026513, 0.0015519, 0.0026049, -0.0006611, 0.0007318
5: 0.0058007, 0.0127578, 0.0056136, 0.0124563, -0.0042963, 0.0047552
6: -0.0016972, 0.0000686, -0.0016207, 0.0001160, -0.0012069, 0.0010904
7: -0.0075289, -0.0029602, -0.0073309, -0.0028374, -0.0031227, 0.0028213
8: -0.0035235, -0.0011209, -0.0034194, -0.0010563, -0.0016422, 0.0014837
9: -0.0005641, 0.0022218, -0.0006390, 0.0021011, -0.0017204, 0.0019042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015014
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015014
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9893532, 0.9932938, 0.9892803, 0.9934104, -0.0027323, 0.0027176
1: -0.0039168, -0.0029350, -0.0039350, -0.0029059, -0.0006808, 0.0006772
2: 0.0054999, 0.0107032, 0.0053459, 0.0107994, -0.0035886, 0.0036079
3: -0.0061448, -0.0037764, -0.0061885, -0.0037063, -0.0016422, 0.0016334
4: 0.0015924, 0.0025995, 0.0015626, 0.0026181, -0.0006946, 0.0006983
5: 0.0058768, 0.0124213, 0.0056831, 0.0125423, -0.0045135, 0.0045378
6: -0.0016118, 0.0000492, -0.0016425, 0.0000984, -0.0011517, 0.0011456
7: -0.0073079, -0.0030103, -0.0073874, -0.0028831, -0.0029799, 0.0029639
8: -0.0034073, -0.0011472, -0.0034491, -0.0010803, -0.0015671, 0.0015587
9: -0.0005336, 0.0020871, -0.0006112, 0.0021355, -0.0018074, 0.0018171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0014766
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0014766
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9892861, 0.9933883, 0.9892604, 0.9934116, -0.0027639, 0.0027405
1: -0.0039336, -0.0029114, -0.0039400, -0.0029056, -0.0006887, 0.0006829
2: 0.0053749, 0.0107918, 0.0053443, 0.0108259, -0.0036188, 0.0036497
3: -0.0061851, -0.0037195, -0.0062006, -0.0037056, -0.0016612, 0.0016471
4: 0.0015682, 0.0026166, 0.0015623, 0.0026232, -0.0007004, 0.0007064
5: 0.0057196, 0.0125328, 0.0056811, 0.0125755, -0.0045515, 0.0045904
6: -0.0016401, 0.0000891, -0.0016510, 0.0000989, -0.0011651, 0.0011552
7: -0.0073811, -0.0029070, -0.0074092, -0.0028817, -0.0030144, 0.0029889
8: -0.0034458, -0.0010929, -0.0034606, -0.0010796, -0.0015853, 0.0015718
9: -0.0005966, 0.0021317, -0.0006120, 0.0021488, -0.0018226, 0.0018382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014138, upper bound: 0.0014766
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014138, upper bound: 0.0014765
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9893321, 0.9934523, -0.0030827, 0.0025220
1: -0.0040199, -0.0029391, -0.0039221, -0.0028955, -0.0007681, 0.0006284
2: 0.0055219, 0.0112494, 0.0052905, 0.0107311, -0.0033303, 0.0040707
3: -0.0063934, -0.0037865, -0.0061574, -0.0036811, -0.0018528, 0.0015158
4: 0.0015966, 0.0027052, 0.0015519, 0.0026049, -0.0006446, 0.0007879
5: 0.0059045, 0.0131082, 0.0056136, 0.0124563, -0.0041886, 0.0051198
6: -0.0017862, 0.0000422, -0.0016207, 0.0001160, -0.0012995, 0.0010631
7: -0.0077590, -0.0030284, -0.0073309, -0.0028374, -0.0033621, 0.0027506
8: -0.0036445, -0.0011568, -0.0034194, -0.0010563, -0.0017681, 0.0014465
9: -0.0005225, 0.0023622, -0.0006390, 0.0021011, -0.0016773, 0.0020502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015105
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015105
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9888133, 0.9932798, 0.9893321, 0.9934523, -0.0032727, 0.0025549
1: -0.0040514, -0.0029385, -0.0039221, -0.0028955, -0.0008155, 0.0006366
2: 0.0055183, 0.0114162, 0.0052905, 0.0107311, -0.0033737, 0.0043216
3: -0.0064693, -0.0037848, -0.0061574, -0.0036811, -0.0019670, 0.0015355
4: 0.0015959, 0.0027375, 0.0015519, 0.0026049, -0.0006530, 0.0008364
5: 0.0059000, 0.0133181, 0.0056136, 0.0124563, -0.0042432, 0.0054354
6: -0.0018394, 0.0000433, -0.0016207, 0.0001160, -0.0013796, 0.0010770
7: -0.0078968, -0.0030255, -0.0073309, -0.0028374, -0.0035694, 0.0027864
8: -0.0037170, -0.0011552, -0.0034194, -0.0010563, -0.0018771, 0.0014654
9: -0.0005243, 0.0024462, -0.0006390, 0.0021011, -0.0016992, 0.0021766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015185
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015185
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9890513, 0.9932044, 0.9892803, 0.9934104, -0.0030613, 0.0026405
1: -0.0039921, -0.0029572, -0.0039350, -0.0029059, -0.0007628, 0.0006579
2: 0.0056178, 0.0111019, 0.0053459, 0.0107994, -0.0034867, 0.0040424
3: -0.0063262, -0.0038301, -0.0061885, -0.0037063, -0.0018399, 0.0015870
4: 0.0016152, 0.0026766, 0.0015626, 0.0026181, -0.0006748, 0.0007824
5: 0.0060252, 0.0129228, 0.0056831, 0.0125423, -0.0043854, 0.0050843
6: -0.0017391, 0.0000116, -0.0016425, 0.0000984, -0.0012904, 0.0011131
7: -0.0076372, -0.0031077, -0.0073874, -0.0028831, -0.0033388, 0.0028798
8: -0.0035805, -0.0011985, -0.0034491, -0.0010803, -0.0017558, 0.0015145
9: -0.0004742, 0.0022879, -0.0006112, 0.0021355, -0.0017561, 0.0020360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0014911
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0014911
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9889811, 0.9933045, 0.9892604, 0.9934116, -0.0030884, 0.0026872
1: -0.0040096, -0.0029323, -0.0039400, -0.0029056, -0.0007695, 0.0006696
2: 0.0054856, 0.0111947, 0.0053443, 0.0108259, -0.0035485, 0.0040781
3: -0.0063685, -0.0037699, -0.0062006, -0.0037056, -0.0018562, 0.0016151
4: 0.0015896, 0.0026946, 0.0015623, 0.0026232, -0.0006868, 0.0007893
5: 0.0058589, 0.0130394, 0.0056811, 0.0125755, -0.0044630, 0.0051292
6: -0.0017687, 0.0000538, -0.0016510, 0.0000989, -0.0013019, 0.0011328
7: -0.0077138, -0.0029985, -0.0074092, -0.0028817, -0.0033683, 0.0029308
8: -0.0036208, -0.0011410, -0.0034606, -0.0010796, -0.0017714, 0.0015413
9: -0.0005408, 0.0023346, -0.0006120, 0.0021488, -0.0017872, 0.0020540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0014912
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0014911
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9890400, 0.9933451, -0.0026489, 0.0029401
1: -0.0039451, -0.0029142, -0.0039949, -0.0029222, -0.0006600, 0.0007326
2: 0.0053900, 0.0108528, 0.0054320, 0.0111169, -0.0038824, 0.0034978
3: -0.0062129, -0.0037264, -0.0063330, -0.0037455, -0.0015920, 0.0017671
4: 0.0015711, 0.0026284, 0.0015792, 0.0026795, -0.0007514, 0.0006770
5: 0.0057386, 0.0126095, 0.0057914, 0.0129416, -0.0048830, 0.0043993
6: -0.0016596, 0.0000843, -0.0017439, 0.0000709, -0.0011166, 0.0012394
7: -0.0074315, -0.0029195, -0.0076496, -0.0029542, -0.0028890, 0.0032066
8: -0.0034723, -0.0010995, -0.0035870, -0.0011177, -0.0015193, 0.0016863
9: -0.0005890, 0.0021624, -0.0005678, 0.0022954, -0.0019554, 0.0017617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0014941
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0014941
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9890400, 0.9933451, -0.0028541, 0.0027244
1: -0.0040199, -0.0029391, -0.0039949, -0.0029222, -0.0007112, 0.0006789
2: 0.0055219, 0.0112494, 0.0054320, 0.0111169, -0.0035976, 0.0037688
3: -0.0063934, -0.0037865, -0.0063330, -0.0037455, -0.0017154, 0.0016375
4: 0.0015966, 0.0027052, 0.0015792, 0.0026795, -0.0006963, 0.0007294
5: 0.0059045, 0.0131082, 0.0057914, 0.0129416, -0.0045248, 0.0047401
6: -0.0017862, 0.0000422, -0.0017439, 0.0000709, -0.0012031, 0.0011484
7: -0.0077590, -0.0030284, -0.0076496, -0.0029542, -0.0031128, 0.0029714
8: -0.0036445, -0.0011568, -0.0035870, -0.0011177, -0.0016370, 0.0015626
9: -0.0005225, 0.0023622, -0.0005678, 0.0022954, -0.0018119, 0.0018982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0015129
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0015128
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9889082, 0.9933501, -0.0026748, 0.0031519
1: -0.0039451, -0.0029142, -0.0040277, -0.0029209, -0.0006665, 0.0007854
2: 0.0053900, 0.0108528, 0.0054254, 0.0112909, -0.0041620, 0.0035320
3: -0.0062129, -0.0037264, -0.0064122, -0.0037425, -0.0016076, 0.0018944
4: 0.0015711, 0.0026284, 0.0015780, 0.0027132, -0.0008055, 0.0006836
5: 0.0057386, 0.0126095, 0.0057832, 0.0131604, -0.0052347, 0.0044423
6: -0.0016596, 0.0000843, -0.0017994, 0.0000730, -0.0011275, 0.0013286
7: -0.0074315, -0.0029195, -0.0077933, -0.0029488, -0.0029172, 0.0034376
8: -0.0034723, -0.0010995, -0.0036625, -0.0011149, -0.0015341, 0.0018078
9: -0.0005890, 0.0021624, -0.0005711, 0.0023831, -0.0020962, 0.0017789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014919
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014919
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9889082, 0.9933501, -0.0028177, 0.0028421
1: -0.0040199, -0.0029391, -0.0040277, -0.0029209, -0.0007021, 0.0007082
2: 0.0055219, 0.0112494, 0.0054254, 0.0112909, -0.0037530, 0.0037207
3: -0.0063934, -0.0037865, -0.0064122, -0.0037425, -0.0016935, 0.0017082
4: 0.0015966, 0.0027052, 0.0015780, 0.0027132, -0.0007264, 0.0007201
5: 0.0059045, 0.0131082, 0.0057832, 0.0131604, -0.0047202, 0.0046797
6: -0.0017862, 0.0000422, -0.0017994, 0.0000730, -0.0011878, 0.0011980
7: -0.0077590, -0.0030284, -0.0077933, -0.0029488, -0.0030731, 0.0030997
8: -0.0036445, -0.0011568, -0.0036625, -0.0011149, -0.0016161, 0.0016301
9: -0.0005225, 0.0023622, -0.0005711, 0.0023831, -0.0018902, 0.0018739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015086
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015086
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9891506, 0.9933396, 0.9890353, 0.9932176, -0.0026822, 0.0029961
1: -0.0039673, -0.0029236, -0.0039961, -0.0029539, -0.0006683, 0.0007466
2: 0.0054393, 0.0109708, 0.0056003, 0.0111231, -0.0039563, 0.0035419
3: -0.0062665, -0.0037489, -0.0063359, -0.0038221, -0.0016121, 0.0018007
4: 0.0015807, 0.0026513, 0.0016118, 0.0026807, -0.0007657, 0.0006855
5: 0.0058007, 0.0127578, 0.0060031, 0.0129494, -0.0049760, 0.0044547
6: -0.0016972, 0.0000686, -0.0017459, 0.0000172, -0.0011307, 0.0012630
7: -0.0075289, -0.0029602, -0.0076547, -0.0030932, -0.0029254, 0.0032677
8: -0.0035235, -0.0011209, -0.0035897, -0.0011908, -0.0015384, 0.0017184
9: -0.0005641, 0.0022218, -0.0004830, 0.0022985, -0.0019926, 0.0017839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0013906
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014626
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9888133, 0.9932798, 0.9890353, 0.9932176, -0.0028287, 0.0028250
1: -0.0040514, -0.0029385, -0.0039961, -0.0029539, -0.0007048, 0.0007039
2: 0.0055183, 0.0114162, 0.0056003, 0.0111231, -0.0037304, 0.0037353
3: -0.0064693, -0.0037848, -0.0063359, -0.0038221, -0.0017001, 0.0016979
4: 0.0015959, 0.0027375, 0.0016118, 0.0026807, -0.0007220, 0.0007230
5: 0.0059000, 0.0133181, 0.0060031, 0.0129494, -0.0046918, 0.0046980
6: -0.0018394, 0.0000433, -0.0017459, 0.0000172, -0.0011924, 0.0011908
7: -0.0078968, -0.0030255, -0.0076547, -0.0030932, -0.0030851, 0.0030810
8: -0.0037170, -0.0011552, -0.0035897, -0.0011908, -0.0016224, 0.0016203
9: -0.0005243, 0.0024462, -0.0004830, 0.0022985, -0.0018788, 0.0018813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0014109
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014911
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891506, 0.9933396, 0.9889337, 0.9933070, -0.0027017, 0.0030057
1: -0.0039673, -0.0029236, -0.0040214, -0.0029317, -0.0006732, 0.0007489
2: 0.0054393, 0.0109708, 0.0054823, 0.0112571, -0.0039690, 0.0035676
3: -0.0062665, -0.0037489, -0.0063969, -0.0037684, -0.0016238, 0.0018065
4: 0.0015807, 0.0026513, 0.0015890, 0.0027067, -0.0007682, 0.0006905
5: 0.0058007, 0.0127578, 0.0058548, 0.0131180, -0.0049920, 0.0044871
6: -0.0016972, 0.0000686, -0.0017887, 0.0000548, -0.0011389, 0.0012670
7: -0.0075289, -0.0029602, -0.0077654, -0.0029958, -0.0029466, 0.0032782
8: -0.0035235, -0.0011209, -0.0036479, -0.0011396, -0.0015496, 0.0017240
9: -0.0005641, 0.0022218, -0.0005424, 0.0023661, -0.0019990, 0.0017968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0013906
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014626
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9888133, 0.9932798, 0.9889337, 0.9933070, -0.0028502, 0.0028331
1: -0.0040514, -0.0029385, -0.0040214, -0.0029317, -0.0007102, 0.0007059
2: 0.0055183, 0.0114162, 0.0054823, 0.0112571, -0.0037411, 0.0037636
3: -0.0064693, -0.0037848, -0.0063969, -0.0037684, -0.0017130, 0.0017028
4: 0.0015959, 0.0027375, 0.0015890, 0.0027067, -0.0007241, 0.0007284
5: 0.0059000, 0.0133181, 0.0058548, 0.0131180, -0.0047053, 0.0047336
6: -0.0018394, 0.0000433, -0.0017887, 0.0000548, -0.0012014, 0.0011943
7: -0.0078968, -0.0030255, -0.0077654, -0.0029958, -0.0031085, 0.0030899
8: -0.0037170, -0.0011552, -0.0036479, -0.0011396, -0.0016347, 0.0016249
9: -0.0005243, 0.0024462, -0.0005424, 0.0023661, -0.0018842, 0.0018955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014109
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014911
time: 0.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014382
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014383
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014387
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014629, upper bound: 0.0014387
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013621, upper bound: 0.0014138
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013621, upper bound: 0.0014137
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014358, upper bound: 0.0014138
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014358, upper bound: 0.0014138
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014422
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014423
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014481
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014533, upper bound: 0.0014481
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013521, upper bound: 0.0014218
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013521, upper bound: 0.0014218
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014218
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014218
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014519
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014519
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014653
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014486, upper bound: 0.0014654
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014303
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014303
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014420
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014892, upper bound: 0.0014420
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0013514
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014236
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0013682
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014236, upper bound: 0.0014493
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0013378
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014011
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0013511
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014626, upper bound: 0.0014218
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015004
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015004
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015014
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0015014
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0014766
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0014766
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014138, upper bound: 0.0014766
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014138, upper bound: 0.0014765
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015105
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015105
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015185
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014307, upper bound: 0.0015185
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0014911
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0013390, upper bound: 0.0014911
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0014912
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0014911
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0014941
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0014941
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0015129
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014324, upper bound: 0.0015128
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014919
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0014919
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015086
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014272, upper bound: 0.0015086
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0013906
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014626
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014010, upper bound: 0.0014109
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014911
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0013906
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014626
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014109
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014911

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9893336, 0.9933008, -0.0025198, 0.0025198
1: -0.0039217, -0.0029332, -0.0039217, -0.0029332, -0.0006279, 0.0006279
2: 0.0054905, 0.0107292, 0.0054905, 0.0107292, -0.0033274, 0.0033274
3: -0.0061566, -0.0037722, -0.0061566, -0.0037722, -0.0015145, 0.0015145
4: 0.0015906, 0.0026045, 0.0015906, 0.0026045, -0.0006440, 0.0006440
5: 0.0058651, 0.0124540, 0.0058651, 0.0124540, -0.0041850, 0.0041850
6: -0.0016201, 0.0000522, -0.0016201, 0.0000522, -0.0010622, 0.0010622
7: -0.0073294, -0.0030025, -0.0073294, -0.0030025, -0.0027482, 0.0027482
8: -0.0034186, -0.0011432, -0.0034186, -0.0011432, -0.0014453, 0.0014453
9: -0.0005383, 0.0021002, -0.0005383, 0.0021002, -0.0016759, 0.0016759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013696, upper bound: 0.0014219
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014437, upper bound: 0.0014219
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9892399, 0.9933769, -0.0025838, 0.0025913
1: -0.0039217, -0.0029332, -0.0039451, -0.0029142, -0.0006438, 0.0006457
2: 0.0054905, 0.0107292, 0.0053900, 0.0108528, -0.0034217, 0.0034119
3: -0.0061566, -0.0037722, -0.0062129, -0.0037264, -0.0015529, 0.0015574
4: 0.0015906, 0.0026045, 0.0015711, 0.0026284, -0.0006623, 0.0006604
5: 0.0058651, 0.0124540, 0.0057386, 0.0126095, -0.0043036, 0.0042912
6: -0.0016201, 0.0000522, -0.0016596, 0.0000843, -0.0010892, 0.0010923
7: -0.0073294, -0.0030025, -0.0074315, -0.0029195, -0.0028180, 0.0028261
8: -0.0034186, -0.0011432, -0.0034723, -0.0010995, -0.0014820, 0.0014862
9: -0.0005383, 0.0021002, -0.0005890, 0.0021624, -0.0017234, 0.0017184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013696, upper bound: 0.0014219
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014437, upper bound: 0.0014219
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9892465, 0.9932592, 0.9893336, 0.9933008, -0.0026372, 0.0024827
1: -0.0039434, -0.0029436, -0.0039217, -0.0029332, -0.0006571, 0.0006186
2: 0.0055454, 0.0108441, 0.0054905, 0.0107292, -0.0032783, 0.0034824
3: -0.0062089, -0.0037972, -0.0061566, -0.0037722, -0.0015850, 0.0014921
4: 0.0016012, 0.0026267, 0.0015906, 0.0026045, -0.0006345, 0.0006740
5: 0.0059341, 0.0125985, 0.0058651, 0.0124540, -0.0041233, 0.0043800
6: -0.0016568, 0.0000347, -0.0016201, 0.0000522, -0.0011117, 0.0010465
7: -0.0074242, -0.0030479, -0.0073294, -0.0030025, -0.0028763, 0.0027077
8: -0.0034685, -0.0011670, -0.0034186, -0.0011432, -0.0015126, 0.0014239
9: -0.0005107, 0.0021580, -0.0005383, 0.0021002, -0.0016511, 0.0017539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013642, upper bound: 0.0014138
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014384, upper bound: 0.0014138
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9892465, 0.9932592, 0.9892399, 0.9933769, -0.0027012, 0.0025541
1: -0.0039434, -0.0029436, -0.0039451, -0.0029142, -0.0006731, 0.0006364
2: 0.0055454, 0.0108441, 0.0053900, 0.0108528, -0.0033726, 0.0035669
3: -0.0062089, -0.0037972, -0.0062129, -0.0037264, -0.0016235, 0.0015351
4: 0.0016012, 0.0026267, 0.0015711, 0.0026284, -0.0006528, 0.0006904
5: 0.0059341, 0.0125985, 0.0057386, 0.0126095, -0.0042419, 0.0044862
6: -0.0016568, 0.0000347, -0.0016596, 0.0000843, -0.0011386, 0.0010766
7: -0.0074242, -0.0030479, -0.0074315, -0.0029195, -0.0029460, 0.0027856
8: -0.0034685, -0.0011670, -0.0034723, -0.0010995, -0.0015493, 0.0014649
9: -0.0005107, 0.0021580, -0.0005890, 0.0021624, -0.0016986, 0.0017965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013642, upper bound: 0.0014137
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014384, upper bound: 0.0014138
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9894422, 0.9932245, 0.9892815, 0.9932573, -0.0025064, 0.0026039
1: -0.0038947, -0.0029522, -0.0039347, -0.0029441, -0.0006245, 0.0006488
2: 0.0055913, 0.0105857, 0.0055481, 0.0107980, -0.0034384, 0.0033096
3: -0.0060913, -0.0038180, -0.0061879, -0.0037984, -0.0015064, 0.0015650
4: 0.0016101, 0.0025767, 0.0016017, 0.0026178, -0.0006655, 0.0006406
5: 0.0059918, 0.0122735, 0.0059374, 0.0125405, -0.0043246, 0.0041626
6: -0.0015743, 0.0000200, -0.0016421, 0.0000338, -0.0010565, 0.0010976
7: -0.0072108, -0.0030858, -0.0073862, -0.0030501, -0.0027335, 0.0028399
8: -0.0033563, -0.0011869, -0.0034485, -0.0011681, -0.0014375, 0.0014935
9: -0.0004876, 0.0020279, -0.0005093, 0.0021348, -0.0017317, 0.0016669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014134
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014138
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9894422, 0.9932245, 0.9891866, 0.9933376, -0.0025773, 0.0026704
1: -0.0038947, -0.0029522, -0.0039584, -0.0029240, -0.0006422, 0.0006654
2: 0.0055913, 0.0105857, 0.0054419, 0.0109232, -0.0035262, 0.0034033
3: -0.0060913, -0.0038180, -0.0062449, -0.0037500, -0.0015490, 0.0016050
4: 0.0016101, 0.0025767, 0.0015812, 0.0026421, -0.0006825, 0.0006587
5: 0.0059918, 0.0122735, 0.0058039, 0.0126980, -0.0044350, 0.0042804
6: -0.0015743, 0.0000200, -0.0016821, 0.0000677, -0.0010864, 0.0011257
7: -0.0072108, -0.0030858, -0.0074896, -0.0029624, -0.0028109, 0.0029124
8: -0.0033563, -0.0011869, -0.0035029, -0.0011220, -0.0014782, 0.0015316
9: -0.0004876, 0.0020279, -0.0005628, 0.0021979, -0.0017760, 0.0017141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014135
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014137
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9893777, 0.9933112, 0.9892616, 0.9932584, -0.0025471, 0.0026328
1: -0.0039107, -0.0029306, -0.0039397, -0.0029438, -0.0006347, 0.0006560
2: 0.0054768, 0.0106709, 0.0055465, 0.0108243, -0.0034766, 0.0033634
3: -0.0061301, -0.0037659, -0.0061999, -0.0037977, -0.0015309, 0.0015824
4: 0.0015879, 0.0025932, 0.0016014, 0.0026229, -0.0006729, 0.0006510
5: 0.0058478, 0.0123807, 0.0059355, 0.0125735, -0.0043726, 0.0042303
6: -0.0016015, 0.0000566, -0.0016505, 0.0000343, -0.0010737, 0.0011098
7: -0.0072812, -0.0029912, -0.0074079, -0.0030488, -0.0027780, 0.0028714
8: -0.0033933, -0.0011372, -0.0034599, -0.0011675, -0.0014609, 0.0015101
9: -0.0005452, 0.0020708, -0.0005101, 0.0021480, -0.0017510, 0.0016940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014135
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014138
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9893777, 0.9933112, 0.9891657, 0.9933388, -0.0026074, 0.0027047
1: -0.0039107, -0.0029306, -0.0039635, -0.0029238, -0.0006497, 0.0006739
2: 0.0054768, 0.0106709, 0.0054404, 0.0109507, -0.0035716, 0.0034430
3: -0.0061301, -0.0037659, -0.0062574, -0.0037494, -0.0015671, 0.0016256
4: 0.0015879, 0.0025932, 0.0015809, 0.0026474, -0.0006913, 0.0006664
5: 0.0058478, 0.0123807, 0.0058020, 0.0127326, -0.0044921, 0.0043304
6: -0.0016015, 0.0000566, -0.0016908, 0.0000682, -0.0010991, 0.0011401
7: -0.0072812, -0.0029912, -0.0075123, -0.0029611, -0.0028437, 0.0029499
8: -0.0033933, -0.0011372, -0.0035148, -0.0011214, -0.0014955, 0.0015513
9: -0.0005452, 0.0020708, -0.0005636, 0.0022117, -0.0017988, 0.0017341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014135
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014138
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9893336, 0.9933008, -0.0028387, 0.0024201
1: -0.0039946, -0.0029617, -0.0039217, -0.0029332, -0.0007073, 0.0006030
2: 0.0056415, 0.0111150, 0.0054905, 0.0107292, -0.0031957, 0.0037484
3: -0.0063322, -0.0038409, -0.0061566, -0.0037722, -0.0017061, 0.0014546
4: 0.0016198, 0.0026792, 0.0015906, 0.0026045, -0.0006185, 0.0007255
5: 0.0060549, 0.0129393, 0.0058651, 0.0124540, -0.0040194, 0.0047145
6: -0.0017433, 0.0000040, -0.0016201, 0.0000522, -0.0011966, 0.0010202
7: -0.0076480, -0.0031272, -0.0073294, -0.0030025, -0.0030960, 0.0026395
8: -0.0035862, -0.0012087, -0.0034186, -0.0011432, -0.0016281, 0.0013881
9: -0.0004623, 0.0022945, -0.0005383, 0.0021002, -0.0016095, 0.0018879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013571, upper bound: 0.0014188
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014287, upper bound: 0.0014188
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9892399, 0.9933769, -0.0029026, 0.0024916
1: -0.0039946, -0.0029617, -0.0039451, -0.0029142, -0.0007233, 0.0006208
2: 0.0056415, 0.0111150, 0.0053900, 0.0108528, -0.0032901, 0.0038329
3: -0.0063322, -0.0038409, -0.0062129, -0.0037264, -0.0017446, 0.0014975
4: 0.0016198, 0.0026792, 0.0015711, 0.0026284, -0.0006368, 0.0007418
5: 0.0060549, 0.0129393, 0.0057386, 0.0126095, -0.0041380, 0.0048208
6: -0.0017433, 0.0000040, -0.0016596, 0.0000843, -0.0012236, 0.0010503
7: -0.0076480, -0.0031272, -0.0074315, -0.0029195, -0.0031657, 0.0027174
8: -0.0035862, -0.0012087, -0.0034723, -0.0010995, -0.0016648, 0.0014290
9: -0.0004623, 0.0022945, -0.0005890, 0.0021624, -0.0016571, 0.0019304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013571, upper bound: 0.0014188
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014287, upper bound: 0.0014188
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9889095, 0.9931933, 0.9893336, 0.9933008, -0.0030478, 0.0024542
1: -0.0040274, -0.0029600, -0.0039217, -0.0029332, -0.0007594, 0.0006115
2: 0.0056324, 0.0112891, 0.0054905, 0.0107292, -0.0032408, 0.0040245
3: -0.0064114, -0.0038367, -0.0061566, -0.0037722, -0.0018318, 0.0014751
4: 0.0016180, 0.0027129, 0.0015906, 0.0026045, -0.0006272, 0.0007789
5: 0.0060435, 0.0131582, 0.0058651, 0.0124540, -0.0040760, 0.0050618
6: -0.0017989, 0.0000069, -0.0016201, 0.0000522, -0.0012847, 0.0010345
7: -0.0077918, -0.0031197, -0.0073294, -0.0030025, -0.0033240, 0.0026767
8: -0.0036618, -0.0012048, -0.0034186, -0.0011432, -0.0017481, 0.0014076
9: -0.0004669, 0.0023822, -0.0005383, 0.0021002, -0.0016322, 0.0020270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013551, upper bound: 0.0014218
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0014218
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9889095, 0.9931933, 0.9892399, 0.9933769, -0.0031117, 0.0025257
1: -0.0040274, -0.0029600, -0.0039451, -0.0029142, -0.0007754, 0.0006293
2: 0.0056324, 0.0112891, 0.0053900, 0.0108528, -0.0033351, 0.0041090
3: -0.0064114, -0.0038367, -0.0062129, -0.0037264, -0.0018702, 0.0015180
4: 0.0016180, 0.0027129, 0.0015711, 0.0026284, -0.0006455, 0.0007953
5: 0.0060435, 0.0131582, 0.0057386, 0.0126095, -0.0041947, 0.0051680
6: -0.0017989, 0.0000069, -0.0016596, 0.0000843, -0.0013117, 0.0010647
7: -0.0077918, -0.0031197, -0.0074315, -0.0029195, -0.0033938, 0.0027546
8: -0.0036618, -0.0012048, -0.0034723, -0.0010995, -0.0017848, 0.0014486
9: -0.0004669, 0.0023822, -0.0005890, 0.0021624, -0.0016797, 0.0020695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013551, upper bound: 0.0014218
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0014218
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9891484, 0.9931169, 0.9892815, 0.9932573, -0.0028189, 0.0025311
1: -0.0039679, -0.0029790, -0.0039347, -0.0029441, -0.0007024, 0.0006307
2: 0.0057333, 0.0109737, 0.0055481, 0.0107980, -0.0033423, 0.0037224
3: -0.0062679, -0.0038827, -0.0061879, -0.0037984, -0.0016943, 0.0015213
4: 0.0016376, 0.0026518, 0.0016017, 0.0026178, -0.0006469, 0.0007205
5: 0.0061705, 0.0127615, 0.0059374, 0.0125405, -0.0042038, 0.0046818
6: -0.0016982, -0.0000253, -0.0016421, 0.0000338, -0.0011883, 0.0010670
7: -0.0075313, -0.0032031, -0.0073862, -0.0030501, -0.0030744, 0.0027606
8: -0.0035248, -0.0012486, -0.0034485, -0.0011681, -0.0016168, 0.0014518
9: -0.0004160, 0.0022233, -0.0005093, 0.0021348, -0.0016834, 0.0018748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014163
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014218
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9891484, 0.9931169, 0.9891866, 0.9933376, -0.0028858, 0.0025976
1: -0.0039679, -0.0029790, -0.0039584, -0.0029240, -0.0007191, 0.0006473
2: 0.0057333, 0.0109737, 0.0054419, 0.0109232, -0.0034302, 0.0038107
3: -0.0062679, -0.0038827, -0.0062449, -0.0037500, -0.0017344, 0.0015613
4: 0.0016376, 0.0026518, 0.0015812, 0.0026421, -0.0006639, 0.0007375
5: 0.0061705, 0.0127615, 0.0058039, 0.0126980, -0.0043142, 0.0047928
6: -0.0016982, -0.0000253, -0.0016821, 0.0000677, -0.0012165, 0.0010950
7: -0.0075313, -0.0032031, -0.0074896, -0.0029624, -0.0031474, 0.0028331
8: -0.0035248, -0.0012486, -0.0035029, -0.0011220, -0.0016552, 0.0014899
9: -0.0004160, 0.0022233, -0.0005628, 0.0021979, -0.0017276, 0.0019192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014163
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014218
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9890829, 0.9932152, 0.9892616, 0.9932584, -0.0028520, 0.0025825
1: -0.0039842, -0.0029546, -0.0039397, -0.0029438, -0.0007106, 0.0006435
2: 0.0056036, 0.0110602, 0.0055465, 0.0108243, -0.0034102, 0.0037660
3: -0.0063072, -0.0038236, -0.0061999, -0.0037977, -0.0017141, 0.0015522
4: 0.0016124, 0.0026686, 0.0016014, 0.0026229, -0.0006600, 0.0007289
5: 0.0060073, 0.0128703, 0.0059355, 0.0125735, -0.0042891, 0.0047367
6: -0.0017258, 0.0000161, -0.0016505, 0.0000343, -0.0012022, 0.0010886
7: -0.0076027, -0.0030959, -0.0074079, -0.0030488, -0.0031105, 0.0028166
8: -0.0035624, -0.0011923, -0.0034599, -0.0011675, -0.0016358, 0.0014812
9: -0.0004814, 0.0022669, -0.0005101, 0.0021480, -0.0017175, 0.0018968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014163
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014218
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9890829, 0.9932152, 0.9891657, 0.9933388, -0.0029122, 0.0026544
1: -0.0039842, -0.0029546, -0.0039635, -0.0029238, -0.0007256, 0.0006614
2: 0.0056036, 0.0110602, 0.0054404, 0.0109507, -0.0035052, 0.0038456
3: -0.0063072, -0.0038236, -0.0062574, -0.0037494, -0.0017503, 0.0015954
4: 0.0016124, 0.0026686, 0.0015809, 0.0026474, -0.0006784, 0.0007443
5: 0.0060073, 0.0128703, 0.0058020, 0.0127326, -0.0044086, 0.0048367
6: -0.0017258, 0.0000161, -0.0016908, 0.0000682, -0.0012276, 0.0011189
7: -0.0076027, -0.0030959, -0.0075123, -0.0029611, -0.0031762, 0.0028951
8: -0.0035624, -0.0011923, -0.0035148, -0.0011214, -0.0016703, 0.0015225
9: -0.0004814, 0.0022669, -0.0005636, 0.0022117, -0.0017654, 0.0019368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014163
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014218
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9890414, 0.9931865, -0.0024201, 0.0028387
1: -0.0039217, -0.0029332, -0.0039946, -0.0029617, -0.0006030, 0.0007073
2: 0.0054905, 0.0107292, 0.0056415, 0.0111150, -0.0037484, 0.0031957
3: -0.0061566, -0.0037722, -0.0063322, -0.0038409, -0.0014546, 0.0017061
4: 0.0015906, 0.0026045, 0.0016198, 0.0026792, -0.0007255, 0.0006185
5: 0.0058651, 0.0124540, 0.0060549, 0.0129393, -0.0047145, 0.0040194
6: -0.0016201, 0.0000522, -0.0017433, 0.0000040, -0.0010202, 0.0011966
7: -0.0073294, -0.0030025, -0.0076480, -0.0031272, -0.0026395, 0.0030960
8: -0.0034186, -0.0011432, -0.0035862, -0.0012087, -0.0013881, 0.0016281
9: -0.0005383, 0.0021002, -0.0004623, 0.0022945, -0.0018879, 0.0016095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013589, upper bound: 0.0014273
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014273
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9889095, 0.9931933, -0.0024542, 0.0030478
1: -0.0039217, -0.0029332, -0.0040274, -0.0029600, -0.0006115, 0.0007594
2: 0.0054905, 0.0107292, 0.0056324, 0.0112891, -0.0040245, 0.0032408
3: -0.0061566, -0.0037722, -0.0064114, -0.0038367, -0.0014751, 0.0018318
4: 0.0015906, 0.0026045, 0.0016180, 0.0027129, -0.0007789, 0.0006272
5: 0.0058651, 0.0124540, 0.0060435, 0.0131582, -0.0050618, 0.0040760
6: -0.0016201, 0.0000522, -0.0017989, 0.0000069, -0.0010345, 0.0012847
7: -0.0073294, -0.0030025, -0.0077918, -0.0031197, -0.0026767, 0.0033240
8: -0.0034186, -0.0011432, -0.0036618, -0.0012048, -0.0014076, 0.0017481
9: -0.0005383, 0.0021002, -0.0004669, 0.0023822, -0.0020270, 0.0016322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013589, upper bound: 0.0014273
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014273
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9890414, 0.9931865, -0.0026247, 0.0026247
1: -0.0039946, -0.0029617, -0.0039946, -0.0029617, -0.0006540, 0.0006540
2: 0.0056415, 0.0111150, 0.0056415, 0.0111150, -0.0034658, 0.0034658
3: -0.0063322, -0.0038409, -0.0063322, -0.0038409, -0.0015775, 0.0015775
4: 0.0016198, 0.0026792, 0.0016198, 0.0026792, -0.0006708, 0.0006708
5: 0.0060549, 0.0129393, 0.0060549, 0.0129393, -0.0043591, 0.0043591
6: -0.0017433, 0.0000040, -0.0017433, 0.0000040, -0.0011064, 0.0011064
7: -0.0076480, -0.0031272, -0.0076480, -0.0031272, -0.0028626, 0.0028626
8: -0.0035862, -0.0012087, -0.0035862, -0.0012087, -0.0015054, 0.0015054
9: -0.0004623, 0.0022945, -0.0004623, 0.0022945, -0.0017456, 0.0017456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0014404
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014231, upper bound: 0.0014404
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9889095, 0.9931933, -0.0025937, 0.0027391
1: -0.0039946, -0.0029617, -0.0040274, -0.0029600, -0.0006463, 0.0006825
2: 0.0056415, 0.0111150, 0.0056324, 0.0112891, -0.0036169, 0.0034250
3: -0.0063322, -0.0038409, -0.0064114, -0.0038367, -0.0015589, 0.0016463
4: 0.0016198, 0.0026792, 0.0016180, 0.0027129, -0.0007000, 0.0006629
5: 0.0060549, 0.0129393, 0.0060435, 0.0131582, -0.0045491, 0.0043077
6: -0.0017433, 0.0000040, -0.0017989, 0.0000069, -0.0010933, 0.0011546
7: -0.0076480, -0.0031272, -0.0077918, -0.0031197, -0.0028288, 0.0029873
8: -0.0035862, -0.0012087, -0.0036618, -0.0012048, -0.0014876, 0.0015710
9: -0.0004623, 0.0022945, -0.0004669, 0.0023822, -0.0018217, 0.0017250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0014404
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014231, upper bound: 0.0014404
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9889396, 0.9932770, -0.0024902, 0.0029269
1: -0.0039217, -0.0029332, -0.0040199, -0.0029391, -0.0006205, 0.0007293
2: 0.0054905, 0.0107292, 0.0055219, 0.0112494, -0.0038649, 0.0032883
3: -0.0061566, -0.0037722, -0.0063934, -0.0037865, -0.0014967, 0.0017591
4: 0.0015906, 0.0026045, 0.0015966, 0.0027052, -0.0007480, 0.0006364
5: 0.0058651, 0.0124540, 0.0059045, 0.0131082, -0.0048610, 0.0041358
6: -0.0016201, 0.0000522, -0.0017862, 0.0000422, -0.0010497, 0.0012338
7: -0.0073294, -0.0030025, -0.0077590, -0.0030284, -0.0027159, 0.0031922
8: -0.0034186, -0.0011432, -0.0036445, -0.0011568, -0.0014283, 0.0016787
9: -0.0005383, 0.0021002, -0.0005225, 0.0023622, -0.0019466, 0.0016562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014013, upper bound: 0.0014046
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014833, upper bound: 0.0014047
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9893336, 0.9933008, 0.9888133, 0.9932798, -0.0025163, 0.0031169
1: -0.0039217, -0.0029332, -0.0040514, -0.0029385, -0.0006270, 0.0007767
2: 0.0054905, 0.0107292, 0.0055183, 0.0114162, -0.0041158, 0.0033227
3: -0.0061566, -0.0037722, -0.0064693, -0.0037848, -0.0015123, 0.0018734
4: 0.0015906, 0.0026045, 0.0015959, 0.0027375, -0.0007966, 0.0006431
5: 0.0058651, 0.0124540, 0.0059000, 0.0133181, -0.0051766, 0.0041791
6: -0.0016201, 0.0000522, -0.0018394, 0.0000433, -0.0010607, 0.0013139
7: -0.0073294, -0.0030025, -0.0078968, -0.0030255, -0.0027443, 0.0033994
8: -0.0034186, -0.0011432, -0.0037170, -0.0011552, -0.0014432, 0.0017877
9: -0.0005383, 0.0021002, -0.0005243, 0.0024462, -0.0020730, 0.0016735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014013, upper bound: 0.0014047
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014833, upper bound: 0.0014046
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9889396, 0.9932770, -0.0026935, 0.0026944
1: -0.0039946, -0.0029617, -0.0040199, -0.0029391, -0.0006711, 0.0006714
2: 0.0056415, 0.0111150, 0.0055219, 0.0112494, -0.0035579, 0.0035567
3: -0.0063322, -0.0038409, -0.0063934, -0.0037865, -0.0016189, 0.0016194
4: 0.0016198, 0.0026792, 0.0015966, 0.0027052, -0.0006886, 0.0006884
5: 0.0060549, 0.0129393, 0.0059045, 0.0131082, -0.0044749, 0.0044734
6: -0.0017433, 0.0000040, -0.0017862, 0.0000422, -0.0011354, 0.0011358
7: -0.0076480, -0.0031272, -0.0077590, -0.0030284, -0.0029376, 0.0029386
8: -0.0035862, -0.0012087, -0.0036445, -0.0011568, -0.0015449, 0.0015454
9: -0.0004623, 0.0022945, -0.0005225, 0.0023622, -0.0017920, 0.0017913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013881, upper bound: 0.0014163
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014625, upper bound: 0.0014163
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9890414, 0.9931865, 0.9888133, 0.9932798, -0.0026628, 0.0028072
1: -0.0039946, -0.0029617, -0.0040514, -0.0029385, -0.0006635, 0.0006995
2: 0.0056415, 0.0111150, 0.0055183, 0.0114162, -0.0037068, 0.0035162
3: -0.0063322, -0.0038409, -0.0064693, -0.0037848, -0.0016004, 0.0016872
4: 0.0016198, 0.0026792, 0.0015959, 0.0027375, -0.0007175, 0.0006806
5: 0.0060549, 0.0129393, 0.0059000, 0.0133181, -0.0046622, 0.0044225
6: -0.0017433, 0.0000040, -0.0018394, 0.0000433, -0.0011225, 0.0011833
7: -0.0076480, -0.0031272, -0.0078968, -0.0030255, -0.0029042, 0.0030616
8: -0.0035862, -0.0012087, -0.0037170, -0.0011552, -0.0015273, 0.0016101
9: -0.0004623, 0.0022945, -0.0005243, 0.0024462, -0.0018670, 0.0017709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013881, upper bound: 0.0014163
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014625, upper bound: 0.0014163
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9892815, 0.9932573, 0.9891484, 0.9931169, -0.0025311, 0.0028189
1: -0.0039347, -0.0029441, -0.0039679, -0.0029790, -0.0006307, 0.0007024
2: 0.0055481, 0.0107980, 0.0057333, 0.0109737, -0.0037224, 0.0033423
3: -0.0061879, -0.0037984, -0.0062679, -0.0038827, -0.0015213, 0.0016943
4: 0.0016017, 0.0026178, 0.0016376, 0.0026518, -0.0007205, 0.0006469
5: 0.0059374, 0.0125405, 0.0061705, 0.0127615, -0.0046818, 0.0042038
6: -0.0016421, 0.0000338, -0.0016982, -0.0000253, -0.0010670, 0.0011883
7: -0.0073862, -0.0030501, -0.0075313, -0.0032031, -0.0027606, 0.0030744
8: -0.0034485, -0.0011681, -0.0035248, -0.0012486, -0.0014518, 0.0016168
9: -0.0005093, 0.0021348, -0.0004160, 0.0022233, -0.0018748, 0.0016834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013999, upper bound: 0.0013106
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013106
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9892616, 0.9932584, 0.9890829, 0.9932152, -0.0025825, 0.0028520
1: -0.0039397, -0.0029438, -0.0039842, -0.0029546, -0.0006435, 0.0007106
2: 0.0055465, 0.0108243, 0.0056036, 0.0110602, -0.0037660, 0.0034102
3: -0.0061999, -0.0037977, -0.0063072, -0.0038236, -0.0015522, 0.0017141
4: 0.0016014, 0.0026229, 0.0016124, 0.0026686, -0.0007289, 0.0006600
5: 0.0059355, 0.0125735, 0.0060073, 0.0128703, -0.0047367, 0.0042891
6: -0.0016505, 0.0000343, -0.0017258, 0.0000161, -0.0010886, 0.0012022
7: -0.0074079, -0.0030488, -0.0076027, -0.0030959, -0.0028166, 0.0031105
8: -0.0034599, -0.0011675, -0.0035624, -0.0011923, -0.0014812, 0.0016358
9: -0.0005101, 0.0021480, -0.0004814, 0.0022669, -0.0018968, 0.0017175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013999, upper bound: 0.0013838
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013838
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9889449, 0.9931908, 0.9891484, 0.9931169, -0.0027050, 0.0026724
1: -0.0040186, -0.0029606, -0.0039679, -0.0029790, -0.0006740, 0.0006659
2: 0.0056358, 0.0112423, 0.0057333, 0.0109737, -0.0035289, 0.0035719
3: -0.0063901, -0.0038383, -0.0062679, -0.0038827, -0.0016258, 0.0016062
4: 0.0016187, 0.0027038, 0.0016376, 0.0026518, -0.0006830, 0.0006913
5: 0.0060478, 0.0130993, 0.0061705, 0.0127615, -0.0044384, 0.0044925
6: -0.0017839, 0.0000058, -0.0016982, -0.0000253, -0.0011402, 0.0011265
7: -0.0077532, -0.0031225, -0.0075313, -0.0032031, -0.0029501, 0.0029146
8: -0.0036415, -0.0012063, -0.0035248, -0.0012486, -0.0015515, 0.0015328
9: -0.0004651, 0.0023586, -0.0004160, 0.0022233, -0.0017773, 0.0017990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013748, upper bound: 0.0013236
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013837, upper bound: 0.0013236
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9889243, 0.9931922, 0.9890829, 0.9932152, -0.0027308, 0.0027017
1: -0.0040238, -0.0029603, -0.0039842, -0.0029546, -0.0006804, 0.0006732
2: 0.0056340, 0.0112698, 0.0056036, 0.0110602, -0.0035676, 0.0036060
3: -0.0064026, -0.0038375, -0.0063072, -0.0038236, -0.0016413, 0.0016238
4: 0.0016183, 0.0027091, 0.0016124, 0.0026686, -0.0006905, 0.0006979
5: 0.0060455, 0.0131339, 0.0060073, 0.0128703, -0.0044871, 0.0045353
6: -0.0017927, 0.0000064, -0.0017258, 0.0000161, -0.0011511, 0.0011389
7: -0.0077759, -0.0031210, -0.0076027, -0.0030959, -0.0029783, 0.0029466
8: -0.0036534, -0.0012055, -0.0035624, -0.0011923, -0.0015663, 0.0015496
9: -0.0004661, 0.0023724, -0.0004814, 0.0022669, -0.0017968, 0.0018161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013748, upper bound: 0.0014073
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013838, upper bound: 0.0014073
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9892815, 0.9932573, 0.9890513, 0.9932044, -0.0025878, 0.0029009
1: -0.0039347, -0.0029441, -0.0039921, -0.0029572, -0.0006448, 0.0007228
2: 0.0055481, 0.0107980, 0.0056178, 0.0111019, -0.0038306, 0.0034171
3: -0.0061879, -0.0037984, -0.0063262, -0.0038301, -0.0015553, 0.0017435
4: 0.0016017, 0.0026178, 0.0016152, 0.0026766, -0.0007414, 0.0006614
5: 0.0059374, 0.0125405, 0.0060252, 0.0129228, -0.0048179, 0.0042978
6: -0.0016421, 0.0000338, -0.0017391, 0.0000116, -0.0010908, 0.0012228
7: -0.0073862, -0.0030501, -0.0076372, -0.0031077, -0.0028223, 0.0031639
8: -0.0034485, -0.0011681, -0.0035805, -0.0011985, -0.0014842, 0.0016639
9: -0.0005093, 0.0021348, -0.0004742, 0.0022879, -0.0019293, 0.0017210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014399, upper bound: 0.0012972
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0012972
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9892616, 0.9932584, 0.9889811, 0.9933045, -0.0026461, 0.0029395
1: -0.0039397, -0.0029438, -0.0040096, -0.0029323, -0.0006593, 0.0007325
2: 0.0055465, 0.0108243, 0.0054856, 0.0111947, -0.0038816, 0.0034942
3: -0.0061999, -0.0037977, -0.0063685, -0.0037699, -0.0015904, 0.0017667
4: 0.0016014, 0.0026229, 0.0015896, 0.0026946, -0.0007513, 0.0006763
5: 0.0059355, 0.0125735, 0.0058589, 0.0130394, -0.0048821, 0.0043948
6: -0.0016505, 0.0000343, -0.0017687, 0.0000538, -0.0011154, 0.0012391
7: -0.0074079, -0.0030488, -0.0077138, -0.0029985, -0.0028860, 0.0032060
8: -0.0034599, -0.0011675, -0.0036208, -0.0011410, -0.0015177, 0.0016860
9: -0.0005101, 0.0021480, -0.0005408, 0.0023346, -0.0019550, 0.0017599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014398, upper bound: 0.0013618
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0013618
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9889449, 0.9931908, 0.9890513, 0.9932044, -0.0027506, 0.0027183
1: -0.0040186, -0.0029606, -0.0039921, -0.0029572, -0.0006854, 0.0006773
2: 0.0056358, 0.0112423, 0.0056178, 0.0111019, -0.0035894, 0.0036322
3: -0.0063901, -0.0038383, -0.0063262, -0.0038301, -0.0016532, 0.0016338
4: 0.0016187, 0.0027038, 0.0016152, 0.0026766, -0.0006947, 0.0007030
5: 0.0060478, 0.0130993, 0.0060252, 0.0129228, -0.0045146, 0.0045683
6: -0.0017839, 0.0000058, -0.0017391, 0.0000116, -0.0011595, 0.0011458
7: -0.0077532, -0.0031225, -0.0076372, -0.0031077, -0.0029999, 0.0029646
8: -0.0036415, -0.0012063, -0.0035805, -0.0011985, -0.0015776, 0.0015591
9: -0.0004651, 0.0023586, -0.0004742, 0.0022879, -0.0018078, 0.0018293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014129, upper bound: 0.0013070
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013070
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9889243, 0.9931922, 0.9889811, 0.9933045, -0.0027973, 0.0027688
1: -0.0040238, -0.0029603, -0.0040096, -0.0029323, -0.0006970, 0.0006899
2: 0.0056340, 0.0112698, 0.0054856, 0.0111947, -0.0036562, 0.0036938
3: -0.0064026, -0.0038375, -0.0063685, -0.0037699, -0.0016813, 0.0016641
4: 0.0016183, 0.0027091, 0.0015896, 0.0026946, -0.0007076, 0.0007149
5: 0.0060455, 0.0131339, 0.0058589, 0.0130394, -0.0045985, 0.0046458
6: -0.0017927, 0.0000064, -0.0017687, 0.0000538, -0.0011792, 0.0011672
7: -0.0077759, -0.0031210, -0.0077138, -0.0029985, -0.0030509, 0.0030198
8: -0.0036534, -0.0012055, -0.0036208, -0.0011410, -0.0016044, 0.0015881
9: -0.0004661, 0.0023724, -0.0005408, 0.0023346, -0.0018414, 0.0018604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014129, upper bound: 0.0013807
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013807
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9893336, 0.9933008, -0.0025913, 0.0025838
1: -0.0039451, -0.0029142, -0.0039217, -0.0029332, -0.0006457, 0.0006438
2: 0.0053900, 0.0108528, 0.0054905, 0.0107292, -0.0034119, 0.0034217
3: -0.0062129, -0.0037264, -0.0061566, -0.0037722, -0.0015574, 0.0015529
4: 0.0015711, 0.0026284, 0.0015906, 0.0026045, -0.0006604, 0.0006623
5: 0.0057386, 0.0126095, 0.0058651, 0.0124540, -0.0042912, 0.0043036
6: -0.0016596, 0.0000843, -0.0016201, 0.0000522, -0.0010923, 0.0010892
7: -0.0074315, -0.0029195, -0.0073294, -0.0030025, -0.0028261, 0.0028180
8: -0.0034723, -0.0010995, -0.0034186, -0.0011432, -0.0014862, 0.0014820
9: -0.0005890, 0.0021624, -0.0005383, 0.0021002, -0.0017184, 0.0017234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013529, upper bound: 0.0014828
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014219, upper bound: 0.0014827
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9892399, 0.9933769, -0.0026036, 0.0026036
1: -0.0039451, -0.0029142, -0.0039451, -0.0029142, -0.0006488, 0.0006488
2: 0.0053900, 0.0108528, 0.0053900, 0.0108528, -0.0034381, 0.0034381
3: -0.0062129, -0.0037264, -0.0062129, -0.0037264, -0.0015649, 0.0015649
4: 0.0015711, 0.0026284, 0.0015711, 0.0026284, -0.0006654, 0.0006654
5: 0.0057386, 0.0126095, 0.0057386, 0.0126095, -0.0043242, 0.0043242
6: -0.0016596, 0.0000843, -0.0016596, 0.0000843, -0.0010975, 0.0010975
7: -0.0074315, -0.0029195, -0.0074315, -0.0029195, -0.0028396, 0.0028396
8: -0.0034723, -0.0010995, -0.0034723, -0.0010995, -0.0014933, 0.0014933
9: -0.0005890, 0.0021624, -0.0005890, 0.0021624, -0.0017316, 0.0017316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013529, upper bound: 0.0014828
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014219, upper bound: 0.0014828
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9891506, 0.9933396, 0.9893336, 0.9933008, -0.0027073, 0.0025538
1: -0.0039673, -0.0029236, -0.0039217, -0.0029332, -0.0006746, 0.0006364
2: 0.0054393, 0.0109708, 0.0054905, 0.0107292, -0.0033723, 0.0035750
3: -0.0062665, -0.0037489, -0.0061566, -0.0037722, -0.0016272, 0.0015349
4: 0.0015807, 0.0026513, 0.0015906, 0.0026045, -0.0006527, 0.0006919
5: 0.0058007, 0.0127578, 0.0058651, 0.0124540, -0.0042415, 0.0044964
6: -0.0016972, 0.0000686, -0.0016201, 0.0000522, -0.0011412, 0.0010765
7: -0.0075289, -0.0029602, -0.0073294, -0.0030025, -0.0029527, 0.0027853
8: -0.0035235, -0.0011209, -0.0034186, -0.0011432, -0.0015528, 0.0014648
9: -0.0005641, 0.0022218, -0.0005383, 0.0021002, -0.0016985, 0.0018006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013471, upper bound: 0.0014766
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0014766
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9891506, 0.9933396, 0.9892399, 0.9933769, -0.0027260, 0.0025682
1: -0.0039673, -0.0029236, -0.0039451, -0.0029142, -0.0006793, 0.0006399
2: 0.0054393, 0.0109708, 0.0053900, 0.0108528, -0.0033913, 0.0035997
3: -0.0062665, -0.0037489, -0.0062129, -0.0037264, -0.0016384, 0.0015436
4: 0.0015807, 0.0026513, 0.0015711, 0.0026284, -0.0006564, 0.0006967
5: 0.0058007, 0.0127578, 0.0057386, 0.0126095, -0.0042654, 0.0045275
6: -0.0016972, 0.0000686, -0.0016596, 0.0000843, -0.0011491, 0.0010826
7: -0.0075289, -0.0029602, -0.0074315, -0.0029195, -0.0029731, 0.0028010
8: -0.0035235, -0.0011209, -0.0034723, -0.0010995, -0.0015635, 0.0014730
9: -0.0005641, 0.0022218, -0.0005890, 0.0021624, -0.0017080, 0.0018130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013471, upper bound: 0.0014766
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0014766
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9893532, 0.9932938, 0.9892815, 0.9932573, -0.0025719, 0.0026496
1: -0.0039168, -0.0029350, -0.0039347, -0.0029441, -0.0006408, 0.0006602
2: 0.0054999, 0.0107032, 0.0055481, 0.0107980, -0.0034988, 0.0033962
3: -0.0061448, -0.0037764, -0.0061879, -0.0037984, -0.0015458, 0.0015925
4: 0.0015924, 0.0025995, 0.0016017, 0.0026178, -0.0006772, 0.0006573
5: 0.0058768, 0.0124213, 0.0059374, 0.0125405, -0.0044005, 0.0042715
6: -0.0016118, 0.0000492, -0.0016421, 0.0000338, -0.0010841, 0.0011169
7: -0.0073079, -0.0030103, -0.0073862, -0.0030501, -0.0028050, 0.0028898
8: -0.0034073, -0.0011472, -0.0034485, -0.0011681, -0.0014751, 0.0015197
9: -0.0005336, 0.0020871, -0.0005093, 0.0021348, -0.0017622, 0.0017105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014756
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014766
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9893532, 0.9932938, 0.9891866, 0.9933376, -0.0025876, 0.0027020
1: -0.0039168, -0.0029350, -0.0039584, -0.0029240, -0.0006448, 0.0006733
2: 0.0054999, 0.0107032, 0.0054419, 0.0109232, -0.0035680, 0.0034169
3: -0.0061448, -0.0037764, -0.0062449, -0.0037500, -0.0015552, 0.0016240
4: 0.0015924, 0.0025995, 0.0015812, 0.0026421, -0.0006906, 0.0006613
5: 0.0058768, 0.0124213, 0.0058039, 0.0126980, -0.0044876, 0.0042975
6: -0.0016118, 0.0000492, -0.0016821, 0.0000677, -0.0010908, 0.0011390
7: -0.0073079, -0.0030103, -0.0074896, -0.0029624, -0.0028221, 0.0029470
8: -0.0034073, -0.0011472, -0.0035029, -0.0011220, -0.0014841, 0.0015498
9: -0.0005336, 0.0020871, -0.0005628, 0.0021979, -0.0017970, 0.0017209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014755
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014766
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9892861, 0.9933883, 0.9892616, 0.9932584, -0.0026151, 0.0027001
1: -0.0039336, -0.0029114, -0.0039397, -0.0029438, -0.0006516, 0.0006728
2: 0.0053749, 0.0107918, 0.0055465, 0.0108243, -0.0035655, 0.0034532
3: -0.0061851, -0.0037195, -0.0061999, -0.0037977, -0.0015717, 0.0016229
4: 0.0015682, 0.0026166, 0.0016014, 0.0026229, -0.0006901, 0.0006684
5: 0.0057196, 0.0125328, 0.0059355, 0.0125735, -0.0044844, 0.0043432
6: -0.0016401, 0.0000891, -0.0016505, 0.0000343, -0.0011023, 0.0011382
7: -0.0073811, -0.0029070, -0.0074079, -0.0030488, -0.0028521, 0.0029449
8: -0.0034458, -0.0010929, -0.0034599, -0.0011675, -0.0014999, 0.0015487
9: -0.0005966, 0.0021317, -0.0005101, 0.0021480, -0.0017958, 0.0017392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014756
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014766
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9892861, 0.9933883, 0.9891657, 0.9933388, -0.0026306, 0.0027242
1: -0.0039336, -0.0029114, -0.0039635, -0.0029238, -0.0006555, 0.0006788
2: 0.0053749, 0.0107918, 0.0054404, 0.0109507, -0.0035972, 0.0034736
3: -0.0061851, -0.0037195, -0.0062574, -0.0037494, -0.0015810, 0.0016373
4: 0.0015682, 0.0026166, 0.0015809, 0.0026474, -0.0006962, 0.0006723
5: 0.0057196, 0.0125328, 0.0058020, 0.0127326, -0.0045244, 0.0043689
6: -0.0016401, 0.0000891, -0.0016908, 0.0000682, -0.0011089, 0.0011483
7: -0.0073811, -0.0029070, -0.0075123, -0.0029611, -0.0028690, 0.0029711
8: -0.0034458, -0.0010929, -0.0035148, -0.0011214, -0.0015088, 0.0015625
9: -0.0005966, 0.0021317, -0.0005636, 0.0022117, -0.0018118, 0.0017495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014756
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014766
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9893336, 0.9933008, -0.0029269, 0.0024902
1: -0.0040199, -0.0029391, -0.0039217, -0.0029332, -0.0007293, 0.0006205
2: 0.0055219, 0.0112494, 0.0054905, 0.0107292, -0.0032883, 0.0038649
3: -0.0063934, -0.0037865, -0.0061566, -0.0037722, -0.0017591, 0.0014967
4: 0.0015966, 0.0027052, 0.0015906, 0.0026045, -0.0006364, 0.0007480
5: 0.0059045, 0.0131082, 0.0058651, 0.0124540, -0.0041358, 0.0048610
6: -0.0017862, 0.0000422, -0.0016201, 0.0000522, -0.0012338, 0.0010497
7: -0.0077590, -0.0030284, -0.0073294, -0.0030025, -0.0031922, 0.0027159
8: -0.0036445, -0.0011568, -0.0034186, -0.0011432, -0.0016787, 0.0014283
9: -0.0005225, 0.0023622, -0.0005383, 0.0021002, -0.0016562, 0.0019466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013406, upper bound: 0.0014857
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014065, upper bound: 0.0014857
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9892399, 0.9933769, -0.0029220, 0.0025034
1: -0.0040199, -0.0029391, -0.0039451, -0.0029142, -0.0007281, 0.0006238
2: 0.0055219, 0.0112494, 0.0053900, 0.0108528, -0.0033057, 0.0038585
3: -0.0063934, -0.0037865, -0.0062129, -0.0037264, -0.0017562, 0.0015046
4: 0.0015966, 0.0027052, 0.0015711, 0.0026284, -0.0006398, 0.0007468
5: 0.0059045, 0.0131082, 0.0057386, 0.0126095, -0.0041577, 0.0048529
6: -0.0017862, 0.0000422, -0.0016596, 0.0000843, -0.0012317, 0.0010553
7: -0.0077590, -0.0030284, -0.0074315, -0.0029195, -0.0031869, 0.0027303
8: -0.0036445, -0.0011568, -0.0034723, -0.0010995, -0.0016759, 0.0014358
9: -0.0005225, 0.0023622, -0.0005890, 0.0021624, -0.0016649, 0.0019433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013406, upper bound: 0.0014857
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014065, upper bound: 0.0014857
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9888133, 0.9932798, 0.9893336, 0.9933008, -0.0031169, 0.0025163
1: -0.0040514, -0.0029385, -0.0039217, -0.0029332, -0.0007767, 0.0006270
2: 0.0055183, 0.0114162, 0.0054905, 0.0107292, -0.0033227, 0.0041158
3: -0.0064693, -0.0037848, -0.0061566, -0.0037722, -0.0018734, 0.0015123
4: 0.0015959, 0.0027375, 0.0015906, 0.0026045, -0.0006431, 0.0007966
5: 0.0059000, 0.0133181, 0.0058651, 0.0124540, -0.0041791, 0.0051766
6: -0.0018394, 0.0000433, -0.0016201, 0.0000522, -0.0013139, 0.0010607
7: -0.0078968, -0.0030255, -0.0073294, -0.0030025, -0.0033994, 0.0027443
8: -0.0037170, -0.0011552, -0.0034186, -0.0011432, -0.0017877, 0.0014432
9: -0.0005243, 0.0024462, -0.0005383, 0.0021002, -0.0016735, 0.0020730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014911
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014047, upper bound: 0.0014911
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9888133, 0.9932798, 0.9892399, 0.9933769, -0.0031361, 0.0025362
1: -0.0040514, -0.0029385, -0.0039451, -0.0029142, -0.0007814, 0.0006320
2: 0.0055183, 0.0114162, 0.0053900, 0.0108528, -0.0033491, 0.0041411
3: -0.0064693, -0.0037848, -0.0062129, -0.0037264, -0.0018849, 0.0015244
4: 0.0015959, 0.0027375, 0.0015711, 0.0026284, -0.0006482, 0.0008015
5: 0.0059000, 0.0133181, 0.0057386, 0.0126095, -0.0042123, 0.0052084
6: -0.0018394, 0.0000433, -0.0016596, 0.0000843, -0.0013220, 0.0010691
7: -0.0078968, -0.0030255, -0.0074315, -0.0029195, -0.0034203, 0.0027661
8: -0.0037170, -0.0011552, -0.0034723, -0.0010995, -0.0017987, 0.0014547
9: -0.0005243, 0.0024462, -0.0005890, 0.0021624, -0.0016868, 0.0020857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014911
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014047, upper bound: 0.0014911
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9890513, 0.9932044, 0.9892815, 0.9932573, -0.0029009, 0.0025878
1: -0.0039921, -0.0029572, -0.0039347, -0.0029441, -0.0007228, 0.0006448
2: 0.0056178, 0.0111019, 0.0055481, 0.0107980, -0.0034171, 0.0038306
3: -0.0063262, -0.0038301, -0.0061879, -0.0037984, -0.0017435, 0.0015553
4: 0.0016152, 0.0026766, 0.0016017, 0.0026178, -0.0006614, 0.0007414
5: 0.0060252, 0.0129228, 0.0059374, 0.0125405, -0.0042978, 0.0048179
6: -0.0017391, 0.0000116, -0.0016421, 0.0000338, -0.0012228, 0.0010908
7: -0.0076372, -0.0031077, -0.0073862, -0.0030501, -0.0031639, 0.0028223
8: -0.0035805, -0.0011985, -0.0034485, -0.0011681, -0.0016639, 0.0014842
9: -0.0004742, 0.0022879, -0.0005093, 0.0021348, -0.0017210, 0.0019293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014813
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014912
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9890513, 0.9932044, 0.9891866, 0.9933376, -0.0028972, 0.0026249
1: -0.0039921, -0.0029572, -0.0039584, -0.0029240, -0.0007219, 0.0006541
2: 0.0056178, 0.0111019, 0.0054419, 0.0109232, -0.0034662, 0.0038257
3: -0.0063262, -0.0038301, -0.0062449, -0.0037500, -0.0017413, 0.0015777
4: 0.0016152, 0.0026766, 0.0015812, 0.0026421, -0.0006709, 0.0007405
5: 0.0060252, 0.0129228, 0.0058039, 0.0126980, -0.0043596, 0.0048117
6: -0.0017391, 0.0000116, -0.0016821, 0.0000677, -0.0012213, 0.0011065
7: -0.0076372, -0.0031077, -0.0074896, -0.0029624, -0.0031598, 0.0028629
8: -0.0035805, -0.0011985, -0.0035029, -0.0011220, -0.0016617, 0.0015056
9: -0.0004742, 0.0022879, -0.0005628, 0.0021979, -0.0017458, 0.0019268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014813
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014911
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9889811, 0.9933045, 0.9892616, 0.9932584, -0.0029395, 0.0026461
1: -0.0040096, -0.0029323, -0.0039397, -0.0029438, -0.0007325, 0.0006593
2: 0.0054856, 0.0111947, 0.0055465, 0.0108243, -0.0034942, 0.0038816
3: -0.0063685, -0.0037699, -0.0061999, -0.0037977, -0.0017667, 0.0015904
4: 0.0015896, 0.0026946, 0.0016014, 0.0026229, -0.0006763, 0.0007513
5: 0.0058589, 0.0130394, 0.0059355, 0.0125735, -0.0043948, 0.0048821
6: -0.0017687, 0.0000538, -0.0016505, 0.0000343, -0.0012391, 0.0011154
7: -0.0077138, -0.0029985, -0.0074079, -0.0030488, -0.0032060, 0.0028860
8: -0.0036208, -0.0011410, -0.0034599, -0.0011675, -0.0016860, 0.0015177
9: -0.0005408, 0.0023346, -0.0005101, 0.0021480, -0.0017599, 0.0019550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014813
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014911
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9889811, 0.9933045, 0.9891657, 0.9933388, -0.0029339, 0.0026709
1: -0.0040096, -0.0029323, -0.0039635, -0.0029238, -0.0007310, 0.0006655
2: 0.0054856, 0.0111947, 0.0054404, 0.0109507, -0.0035269, 0.0038741
3: -0.0063685, -0.0037699, -0.0062574, -0.0037494, -0.0017633, 0.0016053
4: 0.0015896, 0.0026946, 0.0015809, 0.0026474, -0.0006826, 0.0007498
5: 0.0058589, 0.0130394, 0.0058020, 0.0127326, -0.0044359, 0.0048726
6: -0.0017687, 0.0000538, -0.0016908, 0.0000682, -0.0012367, 0.0011259
7: -0.0077138, -0.0029985, -0.0075123, -0.0029611, -0.0031998, 0.0029130
8: -0.0036208, -0.0011410, -0.0035148, -0.0011214, -0.0016827, 0.0015319
9: -0.0005408, 0.0023346, -0.0005636, 0.0022117, -0.0017763, 0.0019512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014813
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014912
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9890414, 0.9931865, -0.0024916, 0.0029026
1: -0.0039451, -0.0029142, -0.0039946, -0.0029617, -0.0006208, 0.0007233
2: 0.0053900, 0.0108528, 0.0056415, 0.0111150, -0.0038329, 0.0032901
3: -0.0062129, -0.0037264, -0.0063322, -0.0038409, -0.0014975, 0.0017446
4: 0.0015711, 0.0026284, 0.0016198, 0.0026792, -0.0007418, 0.0006368
5: 0.0057386, 0.0126095, 0.0060549, 0.0129393, -0.0048208, 0.0041380
6: -0.0016596, 0.0000843, -0.0017433, 0.0000040, -0.0010503, 0.0012236
7: -0.0074315, -0.0029195, -0.0076480, -0.0031272, -0.0027174, 0.0031657
8: -0.0034723, -0.0010995, -0.0035862, -0.0012087, -0.0014291, 0.0016648
9: -0.0005890, 0.0021624, -0.0004623, 0.0022945, -0.0019304, 0.0016571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013429, upper bound: 0.0014682
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014188, upper bound: 0.0014682
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9889396, 0.9932770, -0.0025034, 0.0029220
1: -0.0039451, -0.0029142, -0.0040199, -0.0029391, -0.0006238, 0.0007281
2: 0.0053900, 0.0108528, 0.0055219, 0.0112494, -0.0038585, 0.0033057
3: -0.0062129, -0.0037264, -0.0063934, -0.0037865, -0.0015046, 0.0017562
4: 0.0015711, 0.0026284, 0.0015966, 0.0027052, -0.0007468, 0.0006398
5: 0.0057386, 0.0126095, 0.0059045, 0.0131082, -0.0048529, 0.0041577
6: -0.0016596, 0.0000843, -0.0017862, 0.0000422, -0.0010553, 0.0012317
7: -0.0074315, -0.0029195, -0.0077590, -0.0030284, -0.0027303, 0.0031869
8: -0.0034723, -0.0010995, -0.0036445, -0.0011568, -0.0014358, 0.0016759
9: -0.0005890, 0.0021624, -0.0005225, 0.0023622, -0.0019433, 0.0016649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013429, upper bound: 0.0014682
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014188, upper bound: 0.0014682
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9890414, 0.9931865, -0.0026944, 0.0026935
1: -0.0040199, -0.0029391, -0.0039946, -0.0029617, -0.0006714, 0.0006711
2: 0.0055219, 0.0112494, 0.0056415, 0.0111150, -0.0035567, 0.0035579
3: -0.0063934, -0.0037865, -0.0063322, -0.0038409, -0.0016194, 0.0016189
4: 0.0015966, 0.0027052, 0.0016198, 0.0026792, -0.0006884, 0.0006886
5: 0.0059045, 0.0131082, 0.0060549, 0.0129393, -0.0044734, 0.0044749
6: -0.0017862, 0.0000422, -0.0017433, 0.0000040, -0.0011358, 0.0011354
7: -0.0077590, -0.0030284, -0.0076480, -0.0031272, -0.0029386, 0.0029376
8: -0.0036445, -0.0011568, -0.0035862, -0.0012087, -0.0015454, 0.0015449
9: -0.0005225, 0.0023622, -0.0004623, 0.0022945, -0.0017913, 0.0017920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013357, upper bound: 0.0014857
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014063, upper bound: 0.0014857
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9889396, 0.9932770, -0.0027057, 0.0027057
1: -0.0040199, -0.0029391, -0.0040199, -0.0029391, -0.0006742, 0.0006742
2: 0.0055219, 0.0112494, 0.0055219, 0.0112494, -0.0035728, 0.0035728
3: -0.0063934, -0.0037865, -0.0063934, -0.0037865, -0.0016262, 0.0016262
4: 0.0015966, 0.0027052, 0.0015966, 0.0027052, -0.0006915, 0.0006915
5: 0.0059045, 0.0131082, 0.0059045, 0.0131082, -0.0044937, 0.0044937
6: -0.0017862, 0.0000422, -0.0017862, 0.0000422, -0.0011405, 0.0011405
7: -0.0077590, -0.0030284, -0.0077590, -0.0030284, -0.0029509, 0.0029509
8: -0.0036445, -0.0011568, -0.0036445, -0.0011568, -0.0015519, 0.0015519
9: -0.0005225, 0.0023622, -0.0005225, 0.0023622, -0.0017995, 0.0017995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013357, upper bound: 0.0014857
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014063, upper bound: 0.0014857
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9889095, 0.9931933, -0.0025257, 0.0031117
1: -0.0039451, -0.0029142, -0.0040274, -0.0029600, -0.0006293, 0.0007754
2: 0.0053900, 0.0108528, 0.0056324, 0.0112891, -0.0041090, 0.0033351
3: -0.0062129, -0.0037264, -0.0064114, -0.0038367, -0.0015180, 0.0018702
4: 0.0015711, 0.0026284, 0.0016180, 0.0027129, -0.0007953, 0.0006455
5: 0.0057386, 0.0126095, 0.0060435, 0.0131582, -0.0051680, 0.0041947
6: -0.0016596, 0.0000843, -0.0017989, 0.0000069, -0.0010647, 0.0013117
7: -0.0074315, -0.0029195, -0.0077918, -0.0031197, -0.0027546, 0.0033938
8: -0.0034723, -0.0010995, -0.0036618, -0.0012048, -0.0014486, 0.0017848
9: -0.0005890, 0.0021624, -0.0004669, 0.0023822, -0.0020695, 0.0016797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014661
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014218, upper bound: 0.0014661
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9892399, 0.9933769, 0.9888133, 0.9932798, -0.0025362, 0.0031360
1: -0.0039451, -0.0029142, -0.0040514, -0.0029385, -0.0006320, 0.0007814
2: 0.0053900, 0.0108528, 0.0055183, 0.0114162, -0.0041411, 0.0033491
3: -0.0062129, -0.0037264, -0.0064693, -0.0037848, -0.0015244, 0.0018849
4: 0.0015711, 0.0026284, 0.0015959, 0.0027375, -0.0008015, 0.0006482
5: 0.0057386, 0.0126095, 0.0059000, 0.0133181, -0.0052084, 0.0042123
6: -0.0016596, 0.0000843, -0.0018394, 0.0000433, -0.0010691, 0.0013220
7: -0.0074315, -0.0029195, -0.0078968, -0.0030255, -0.0027661, 0.0034203
8: -0.0034723, -0.0010995, -0.0037170, -0.0011552, -0.0014547, 0.0017987
9: -0.0005890, 0.0021624, -0.0005243, 0.0024462, -0.0020857, 0.0016868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014660
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014218, upper bound: 0.0014661
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9889095, 0.9931933, -0.0026634, 0.0028079
1: -0.0040199, -0.0029391, -0.0040274, -0.0029600, -0.0006637, 0.0006996
2: 0.0055219, 0.0112494, 0.0056324, 0.0112891, -0.0037078, 0.0035170
3: -0.0063934, -0.0037865, -0.0064114, -0.0038367, -0.0016008, 0.0016876
4: 0.0015966, 0.0027052, 0.0016180, 0.0027129, -0.0007176, 0.0006807
5: 0.0059045, 0.0131082, 0.0060435, 0.0131582, -0.0046634, 0.0044235
6: -0.0017862, 0.0000422, -0.0017989, 0.0000069, -0.0011227, 0.0011836
7: -0.0077590, -0.0030284, -0.0077918, -0.0031197, -0.0029049, 0.0030624
8: -0.0036445, -0.0011568, -0.0036618, -0.0012048, -0.0015276, 0.0016105
9: -0.0005225, 0.0023622, -0.0004669, 0.0023822, -0.0018674, 0.0017714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0014814
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014814
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9889396, 0.9932770, 0.9888133, 0.9932798, -0.0026733, 0.0028262
1: -0.0040199, -0.0029391, -0.0040514, -0.0029385, -0.0006661, 0.0007042
2: 0.0055219, 0.0112494, 0.0055183, 0.0114162, -0.0037320, 0.0035301
3: -0.0063934, -0.0037865, -0.0064693, -0.0037848, -0.0016067, 0.0016986
4: 0.0015966, 0.0027052, 0.0015959, 0.0027375, -0.0007223, 0.0006832
5: 0.0059045, 0.0131082, 0.0059000, 0.0133181, -0.0046938, 0.0044399
6: -0.0017862, 0.0000422, -0.0018394, 0.0000433, -0.0011269, 0.0011913
7: -0.0077590, -0.0030284, -0.0078968, -0.0030255, -0.0029156, 0.0030824
8: -0.0036445, -0.0011568, -0.0037170, -0.0011552, -0.0015333, 0.0016210
9: -0.0005225, 0.0023622, -0.0005243, 0.0024462, -0.0018796, 0.0017779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0014813
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014814
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9891866, 0.9933376, 0.9891484, 0.9931169, -0.0025976, 0.0028858
1: -0.0039584, -0.0029240, -0.0039679, -0.0029790, -0.0006473, 0.0007191
2: 0.0054419, 0.0109232, 0.0057333, 0.0109737, -0.0038107, 0.0034302
3: -0.0062449, -0.0037500, -0.0062679, -0.0038827, -0.0015613, 0.0017344
4: 0.0015812, 0.0026421, 0.0016376, 0.0026518, -0.0007375, 0.0006639
5: 0.0058039, 0.0126980, 0.0061705, 0.0127615, -0.0047928, 0.0043142
6: -0.0016821, 0.0000677, -0.0016982, -0.0000253, -0.0010950, 0.0012165
7: -0.0074896, -0.0029624, -0.0075313, -0.0032031, -0.0028331, 0.0031474
8: -0.0035029, -0.0011220, -0.0035248, -0.0012486, -0.0014899, 0.0016552
9: -0.0005628, 0.0021979, -0.0004160, 0.0022233, -0.0019192, 0.0017276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0013495
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013807, upper bound: 0.0013495
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9891657, 0.9933388, 0.9890829, 0.9932152, -0.0026544, 0.0029122
1: -0.0039635, -0.0029238, -0.0039842, -0.0029546, -0.0006614, 0.0007256
2: 0.0054404, 0.0109507, 0.0056036, 0.0110602, -0.0038456, 0.0035052
3: -0.0062574, -0.0037494, -0.0063072, -0.0038236, -0.0015954, 0.0017503
4: 0.0015809, 0.0026474, 0.0016124, 0.0026686, -0.0007443, 0.0006784
5: 0.0058020, 0.0127326, 0.0060073, 0.0128703, -0.0048367, 0.0044086
6: -0.0016908, 0.0000682, -0.0017258, 0.0000161, -0.0011189, 0.0012276
7: -0.0075123, -0.0029611, -0.0076027, -0.0030959, -0.0028951, 0.0031762
8: -0.0035148, -0.0011214, -0.0035624, -0.0011923, -0.0015225, 0.0016703
9: -0.0005636, 0.0022117, -0.0004814, 0.0022669, -0.0019368, 0.0017654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0014227
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013807, upper bound: 0.0014226
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9888487, 0.9932772, 0.9891484, 0.9931169, -0.0027697, 0.0027372
1: -0.0040425, -0.0029391, -0.0039679, -0.0029790, -0.0006901, 0.0006820
2: 0.0055216, 0.0113694, 0.0057333, 0.0109737, -0.0036145, 0.0036574
3: -0.0064480, -0.0037863, -0.0062679, -0.0038827, -0.0016647, 0.0016452
4: 0.0015966, 0.0027284, 0.0016376, 0.0026518, -0.0006996, 0.0007079
5: 0.0059042, 0.0132591, 0.0061705, 0.0127615, -0.0045461, 0.0046001
6: -0.0018245, 0.0000423, -0.0016982, -0.0000253, -0.0011675, 0.0011538
7: -0.0078581, -0.0030282, -0.0075313, -0.0032031, -0.0030208, 0.0029853
8: -0.0036966, -0.0011567, -0.0035248, -0.0012486, -0.0015886, 0.0015700
9: -0.0005226, 0.0024226, -0.0004160, 0.0022233, -0.0018204, 0.0018421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0013648
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0013648
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9888287, 0.9932786, 0.9890829, 0.9932152, -0.0028008, 0.0027561
1: -0.0040475, -0.0029387, -0.0039842, -0.0029546, -0.0006979, 0.0006867
2: 0.0055198, 0.0113958, 0.0056036, 0.0110602, -0.0036394, 0.0036984
3: -0.0064600, -0.0037855, -0.0063072, -0.0038236, -0.0016834, 0.0016565
4: 0.0015962, 0.0027335, 0.0016124, 0.0026686, -0.0007044, 0.0007158
5: 0.0059019, 0.0132924, 0.0060073, 0.0128703, -0.0045774, 0.0046517
6: -0.0018329, 0.0000429, -0.0017258, 0.0000161, -0.0011806, 0.0011618
7: -0.0078799, -0.0030267, -0.0076027, -0.0030959, -0.0030547, 0.0030059
8: -0.0037081, -0.0011559, -0.0035624, -0.0011923, -0.0016064, 0.0015808
9: -0.0005236, 0.0024359, -0.0004814, 0.0022669, -0.0018330, 0.0018627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0014474
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0014474
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9891866, 0.9933376, 0.9890513, 0.9932044, -0.0026249, 0.0028972
1: -0.0039584, -0.0029240, -0.0039921, -0.0029572, -0.0006541, 0.0007219
2: 0.0054419, 0.0109232, 0.0056178, 0.0111019, -0.0038257, 0.0034662
3: -0.0062449, -0.0037500, -0.0063262, -0.0038301, -0.0015777, 0.0017413
4: 0.0015812, 0.0026421, 0.0016152, 0.0026766, -0.0007405, 0.0006709
5: 0.0058039, 0.0126980, 0.0060252, 0.0129228, -0.0048117, 0.0043596
6: -0.0016821, 0.0000677, -0.0017391, 0.0000116, -0.0011065, 0.0012213
7: -0.0074896, -0.0029624, -0.0076372, -0.0031077, -0.0028629, 0.0031598
8: -0.0035029, -0.0011220, -0.0035805, -0.0011985, -0.0015056, 0.0016617
9: -0.0005628, 0.0021979, -0.0004742, 0.0022879, -0.0019268, 0.0017458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0013495
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013806, upper bound: 0.0013495
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9891657, 0.9933388, 0.9889811, 0.9933045, -0.0026709, 0.0029339
1: -0.0039635, -0.0029238, -0.0040096, -0.0029323, -0.0006655, 0.0007310
2: 0.0054404, 0.0109507, 0.0054856, 0.0111947, -0.0038741, 0.0035269
3: -0.0062574, -0.0037494, -0.0063685, -0.0037699, -0.0016053, 0.0017633
4: 0.0015809, 0.0026474, 0.0015896, 0.0026946, -0.0007498, 0.0006826
5: 0.0058020, 0.0127326, 0.0058589, 0.0130394, -0.0048726, 0.0044359
6: -0.0016908, 0.0000682, -0.0017687, 0.0000538, -0.0011259, 0.0012367
7: -0.0075123, -0.0029611, -0.0077138, -0.0029985, -0.0029130, 0.0031998
8: -0.0035148, -0.0011214, -0.0036208, -0.0011410, -0.0015319, 0.0016827
9: -0.0005636, 0.0022117, -0.0005408, 0.0023346, -0.0019512, 0.0017763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0014226
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013806, upper bound: 0.0014226
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9888487, 0.9932772, 0.9890513, 0.9932044, -0.0028002, 0.0027302
1: -0.0040425, -0.0029391, -0.0039921, -0.0029572, -0.0006977, 0.0006803
2: 0.0055216, 0.0113694, 0.0056178, 0.0111019, -0.0036052, 0.0036977
3: -0.0064480, -0.0037863, -0.0063262, -0.0038301, -0.0016830, 0.0016409
4: 0.0015966, 0.0027284, 0.0016152, 0.0026766, -0.0006978, 0.0007157
5: 0.0059042, 0.0132591, 0.0060252, 0.0129228, -0.0045344, 0.0046507
6: -0.0018245, 0.0000423, -0.0017391, 0.0000116, -0.0011804, 0.0011509
7: -0.0078581, -0.0030282, -0.0076372, -0.0031077, -0.0030541, 0.0029777
8: -0.0036966, -0.0011567, -0.0035805, -0.0011985, -0.0016061, 0.0015659
9: -0.0005226, 0.0024226, -0.0004742, 0.0022879, -0.0018158, 0.0018623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0013648
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0013648
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9888287, 0.9932786, 0.9889811, 0.9933045, -0.0028189, 0.0027777
1: -0.0040475, -0.0029387, -0.0040096, -0.0029323, -0.0007024, 0.0006921
2: 0.0055198, 0.0113958, 0.0054856, 0.0111947, -0.0036679, 0.0037223
3: -0.0064600, -0.0037855, -0.0063685, -0.0037699, -0.0016942, 0.0016695
4: 0.0015962, 0.0027335, 0.0015896, 0.0026946, -0.0007099, 0.0007204
5: 0.0059019, 0.0132924, 0.0058589, 0.0130394, -0.0046133, 0.0046817
6: -0.0018329, 0.0000429, -0.0017687, 0.0000538, -0.0011883, 0.0011709
7: -0.0078799, -0.0030267, -0.0077138, -0.0029985, -0.0030744, 0.0030295
8: -0.0037081, -0.0011559, -0.0036208, -0.0011410, -0.0016168, 0.0015932
9: -0.0005236, 0.0024359, -0.0005408, 0.0023346, -0.0018474, 0.0018748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0014474
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0014474
time: 0.96 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.39 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013696, upper bound: 0.0014219
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014437, upper bound: 0.0014219
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013696, upper bound: 0.0014219
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014437, upper bound: 0.0014219
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013642, upper bound: 0.0014138
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014384, upper bound: 0.0014138
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013642, upper bound: 0.0014137
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014384, upper bound: 0.0014138
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014134
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014138
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014135
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013609, upper bound: 0.0014137
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014135
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014138
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014135
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014353, upper bound: 0.0014138
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013571, upper bound: 0.0014188
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014287, upper bound: 0.0014188
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013571, upper bound: 0.0014188
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014287, upper bound: 0.0014188
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013551, upper bound: 0.0014218
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0014218
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013551, upper bound: 0.0014218
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0014218
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014163
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014218
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014163
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014218
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014163
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014218
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014163
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014232, upper bound: 0.0014218
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013589, upper bound: 0.0014273
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014273
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013589, upper bound: 0.0014273
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014411, upper bound: 0.0014273
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0014404
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014231, upper bound: 0.0014404
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0014404
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014231, upper bound: 0.0014404
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014013, upper bound: 0.0014046
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014833, upper bound: 0.0014047
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014013, upper bound: 0.0014047
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014833, upper bound: 0.0014046
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013881, upper bound: 0.0014163
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014625, upper bound: 0.0014163
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013881, upper bound: 0.0014163
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014625, upper bound: 0.0014163
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013999, upper bound: 0.0013106
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013106
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013999, upper bound: 0.0013838
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013838
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013748, upper bound: 0.0013236
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013837, upper bound: 0.0013236
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013748, upper bound: 0.0014073
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013838, upper bound: 0.0014073
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014399, upper bound: 0.0012972
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0012972
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014398, upper bound: 0.0013618
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0013618
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014129, upper bound: 0.0013070
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013070
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014129, upper bound: 0.0013807
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013807
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013529, upper bound: 0.0014828
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014219, upper bound: 0.0014827
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013529, upper bound: 0.0014828
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014219, upper bound: 0.0014828
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013471, upper bound: 0.0014766
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0014766
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013471, upper bound: 0.0014766
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0014766
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014756
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014766
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014755
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014766
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014756
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014766
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014756
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014766
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013406, upper bound: 0.0014857
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014065, upper bound: 0.0014857
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013406, upper bound: 0.0014857
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014065, upper bound: 0.0014857
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014911
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014047, upper bound: 0.0014911
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014911
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014047, upper bound: 0.0014911
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014813
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014912
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014813
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014911
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014813
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014911
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014813
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014912
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013429, upper bound: 0.0014682
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014188, upper bound: 0.0014682
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013429, upper bound: 0.0014682
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014188, upper bound: 0.0014682
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013357, upper bound: 0.0014857
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014063, upper bound: 0.0014857
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013357, upper bound: 0.0014857
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014063, upper bound: 0.0014857
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014661
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014218, upper bound: 0.0014661
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014660
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014218, upper bound: 0.0014661
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0014814
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014814
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0014813
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014814
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0013495
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013807, upper bound: 0.0013495
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0014227
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013807, upper bound: 0.0014226
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0013648
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0013648
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0014474
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0014474
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0013495
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013806, upper bound: 0.0013495
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0014226
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013806, upper bound: 0.0014226
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0013648
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0013648
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0014474
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0014474

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9893686, 0.9932994, -0.0023802, 0.0024548
1: -0.0038941, -0.0029555, -0.0039130, -0.0029336, -0.0005931, 0.0006117
2: 0.0056086, 0.0105828, 0.0054924, 0.0106830, -0.0032416, 0.0031430
3: -0.0060900, -0.0038259, -0.0061355, -0.0037730, -0.0014306, 0.0014754
4: 0.0016134, 0.0025762, 0.0015909, 0.0025956, -0.0006274, 0.0006083
5: 0.0060135, 0.0122699, 0.0058674, 0.0123958, -0.0040770, 0.0039531
6: -0.0015734, 0.0000145, -0.0016054, 0.0000516, -0.0010033, 0.0010348
7: -0.0072085, -0.0031000, -0.0072912, -0.0030041, -0.0025959, 0.0026773
8: -0.0033550, -0.0011944, -0.0033985, -0.0011440, -0.0013652, 0.0014080
9: -0.0004789, 0.0020265, -0.0005374, 0.0020769, -0.0016326, 0.0015830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013287, upper bound: 0.0013951
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013287, upper bound: 0.0014052
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9893485, 0.9933001, -0.0024597, 0.0024890
1: -0.0039097, -0.0029337, -0.0039180, -0.0029334, -0.0006129, 0.0006202
2: 0.0054932, 0.0106653, 0.0054913, 0.0107095, -0.0032867, 0.0032479
3: -0.0061275, -0.0037734, -0.0061476, -0.0037725, -0.0014783, 0.0014960
4: 0.0015911, 0.0025921, 0.0015907, 0.0026007, -0.0006361, 0.0006286
5: 0.0058684, 0.0123736, 0.0058661, 0.0124292, -0.0041338, 0.0040851
6: -0.0015997, 0.0000514, -0.0016138, 0.0000520, -0.0010368, 0.0010492
7: -0.0072766, -0.0030047, -0.0073131, -0.0030032, -0.0026826, 0.0027146
8: -0.0033908, -0.0011443, -0.0034100, -0.0011435, -0.0014108, 0.0014276
9: -0.0005370, 0.0020680, -0.0005379, 0.0020902, -0.0016553, 0.0016358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014051, upper bound: 0.0013951
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014051, upper bound: 0.0014051
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9892759, 0.9933755, -0.0024452, 0.0025240
1: -0.0038941, -0.0029555, -0.0039361, -0.0029146, -0.0006093, 0.0006289
2: 0.0056086, 0.0105828, 0.0053919, 0.0108053, -0.0033329, 0.0032288
3: -0.0060900, -0.0038259, -0.0061912, -0.0037273, -0.0014696, 0.0015170
4: 0.0016134, 0.0025762, 0.0015715, 0.0026192, -0.0006451, 0.0006249
5: 0.0060135, 0.0122699, 0.0057410, 0.0125497, -0.0041919, 0.0040610
6: -0.0015734, 0.0000145, -0.0016444, 0.0000837, -0.0010307, 0.0010640
7: -0.0072085, -0.0031000, -0.0073922, -0.0029211, -0.0026668, 0.0027528
8: -0.0033550, -0.0011944, -0.0034517, -0.0011003, -0.0014024, 0.0014477
9: -0.0004789, 0.0020265, -0.0005880, 0.0021385, -0.0016786, 0.0016262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013690, upper bound: 0.0013733
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013691, upper bound: 0.0013830
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9892550, 0.9933764, -0.0025157, 0.0025612
1: -0.0039097, -0.0029337, -0.0039413, -0.0029144, -0.0006268, 0.0006382
2: 0.0054932, 0.0106653, 0.0053908, 0.0108330, -0.0033820, 0.0033219
3: -0.0061275, -0.0037734, -0.0062038, -0.0037268, -0.0015120, 0.0015393
4: 0.0015911, 0.0025921, 0.0015713, 0.0026246, -0.0006546, 0.0006429
5: 0.0058684, 0.0123736, 0.0057396, 0.0125845, -0.0042537, 0.0041781
6: -0.0015997, 0.0000514, -0.0016533, 0.0000840, -0.0010604, 0.0010796
7: -0.0072766, -0.0030047, -0.0074151, -0.0029202, -0.0027437, 0.0027933
8: -0.0033908, -0.0011443, -0.0034637, -0.0010998, -0.0014429, 0.0014690
9: -0.0005370, 0.0020680, -0.0005885, 0.0021524, -0.0017034, 0.0016731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014443, upper bound: 0.0013733
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014443, upper bound: 0.0013830
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893543, 0.9931653, 0.9893686, 0.9932994, -0.0024988, 0.0024183
1: -0.0039166, -0.0029670, -0.0039130, -0.0029336, -0.0006226, 0.0006026
2: 0.0056695, 0.0107018, 0.0054924, 0.0106830, -0.0031934, 0.0032996
3: -0.0061441, -0.0038536, -0.0061355, -0.0037730, -0.0015018, 0.0014535
4: 0.0016252, 0.0025992, 0.0015909, 0.0025956, -0.0006181, 0.0006386
5: 0.0060901, 0.0124195, 0.0058674, 0.0123958, -0.0040164, 0.0041500
6: -0.0016114, -0.0000049, -0.0016054, 0.0000516, -0.0010533, 0.0010194
7: -0.0073067, -0.0031503, -0.0072912, -0.0030041, -0.0027252, 0.0026375
8: -0.0034067, -0.0012209, -0.0033985, -0.0011440, -0.0014332, 0.0013870
9: -0.0004482, 0.0020864, -0.0005374, 0.0020769, -0.0016083, 0.0016618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0013897
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0013995
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9892952, 0.9932564, 0.9893485, 0.9933001, -0.0025796, 0.0024526
1: -0.0039313, -0.0029443, -0.0039180, -0.0029334, -0.0006428, 0.0006111
2: 0.0055491, 0.0107798, 0.0054913, 0.0107095, -0.0032387, 0.0034064
3: -0.0061796, -0.0037988, -0.0061476, -0.0037725, -0.0015504, 0.0014741
4: 0.0016019, 0.0026143, 0.0015907, 0.0026007, -0.0006268, 0.0006593
5: 0.0059388, 0.0125176, 0.0058661, 0.0124292, -0.0040734, 0.0042844
6: -0.0016363, 0.0000335, -0.0016138, 0.0000520, -0.0010874, 0.0010339
7: -0.0073711, -0.0030509, -0.0073131, -0.0030032, -0.0028135, 0.0026749
8: -0.0034405, -0.0011686, -0.0034100, -0.0011435, -0.0014796, 0.0014067
9: -0.0005088, 0.0021256, -0.0005379, 0.0020902, -0.0016312, 0.0017156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014017, upper bound: 0.0013897
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014017, upper bound: 0.0013995
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9893543, 0.9931653, 0.9892759, 0.9933755, -0.0025637, 0.0024875
1: -0.0039166, -0.0029670, -0.0039361, -0.0029146, -0.0006388, 0.0006198
2: 0.0056695, 0.0107018, 0.0053919, 0.0108053, -0.0032847, 0.0033854
3: -0.0061441, -0.0038536, -0.0061912, -0.0037273, -0.0015409, 0.0014951
4: 0.0016252, 0.0025992, 0.0015715, 0.0026192, -0.0006357, 0.0006552
5: 0.0060901, 0.0124195, 0.0057410, 0.0125497, -0.0041313, 0.0042579
6: -0.0016114, -0.0000049, -0.0016444, 0.0000837, -0.0010807, 0.0010486
7: -0.0073067, -0.0031503, -0.0073922, -0.0029211, -0.0027961, 0.0027130
8: -0.0034067, -0.0012209, -0.0034517, -0.0011003, -0.0014704, 0.0014267
9: -0.0004482, 0.0020864, -0.0005880, 0.0021385, -0.0016544, 0.0017051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013663, upper bound: 0.0013663
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013663, upper bound: 0.0013778
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9892952, 0.9932564, 0.9892550, 0.9933764, -0.0026357, 0.0025248
1: -0.0039313, -0.0029443, -0.0039413, -0.0029144, -0.0006567, 0.0006291
2: 0.0055491, 0.0107798, 0.0053908, 0.0108330, -0.0033340, 0.0034804
3: -0.0061796, -0.0037988, -0.0062038, -0.0037268, -0.0015841, 0.0015175
4: 0.0016019, 0.0026143, 0.0015713, 0.0026246, -0.0006453, 0.0006736
5: 0.0059388, 0.0125176, 0.0057396, 0.0125845, -0.0041933, 0.0043774
6: -0.0016363, 0.0000335, -0.0016533, 0.0000840, -0.0011110, 0.0010643
7: -0.0073711, -0.0030509, -0.0074151, -0.0029202, -0.0028746, 0.0027537
8: -0.0034405, -0.0011686, -0.0034637, -0.0010998, -0.0015117, 0.0014481
9: -0.0005088, 0.0021256, -0.0005885, 0.0021524, -0.0016792, 0.0017529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014410, upper bound: 0.0013663
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014410, upper bound: 0.0013779
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9892815, 0.9932573, -0.0023494, 0.0025722
1: -0.0038941, -0.0029555, -0.0039347, -0.0029441, -0.0005854, 0.0006409
2: 0.0056086, 0.0105828, 0.0055481, 0.0107980, -0.0033966, 0.0031024
3: -0.0060900, -0.0038259, -0.0061879, -0.0037984, -0.0014121, 0.0015460
4: 0.0016134, 0.0025762, 0.0016017, 0.0026178, -0.0006574, 0.0006005
5: 0.0060135, 0.0122699, 0.0059374, 0.0125405, -0.0042720, 0.0039020
6: -0.0015734, 0.0000145, -0.0016421, 0.0000338, -0.0009904, 0.0010843
7: -0.0072085, -0.0031000, -0.0073862, -0.0030501, -0.0025624, 0.0028054
8: -0.0033550, -0.0011944, -0.0034485, -0.0011681, -0.0013475, 0.0014753
9: -0.0004789, 0.0020265, -0.0005093, 0.0021348, -0.0017107, 0.0015625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013891
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013989
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9893543, 0.9931653, 0.9892815, 0.9932573, -0.0025023, 0.0025398
1: -0.0039166, -0.0029670, -0.0039347, -0.0029441, -0.0006235, 0.0006329
2: 0.0056695, 0.0107018, 0.0055481, 0.0107980, -0.0033538, 0.0033043
3: -0.0061441, -0.0038536, -0.0061879, -0.0037984, -0.0015040, 0.0015265
4: 0.0016252, 0.0025992, 0.0016017, 0.0026178, -0.0006491, 0.0006395
5: 0.0060901, 0.0124195, 0.0059374, 0.0125405, -0.0042182, 0.0041559
6: -0.0016114, -0.0000049, -0.0016421, 0.0000338, -0.0010548, 0.0010706
7: -0.0073067, -0.0031503, -0.0073862, -0.0030501, -0.0027291, 0.0027701
8: -0.0034067, -0.0012209, -0.0034485, -0.0011681, -0.0014352, 0.0014567
9: -0.0004482, 0.0020864, -0.0005093, 0.0021348, -0.0016892, 0.0016642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013897
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013995
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9891866, 0.9933376, -0.0024152, 0.0026387
1: -0.0038941, -0.0029555, -0.0039584, -0.0029240, -0.0006018, 0.0006575
2: 0.0056086, 0.0105828, 0.0054419, 0.0109232, -0.0034844, 0.0031893
3: -0.0060900, -0.0038259, -0.0062449, -0.0037500, -0.0014516, 0.0015860
4: 0.0016134, 0.0025762, 0.0015812, 0.0026421, -0.0006744, 0.0006173
5: 0.0060135, 0.0122699, 0.0058039, 0.0126980, -0.0043825, 0.0040112
6: -0.0015734, 0.0000145, -0.0016821, 0.0000677, -0.0010181, 0.0011123
7: -0.0072085, -0.0031000, -0.0074896, -0.0029624, -0.0026341, 0.0028779
8: -0.0033550, -0.0011944, -0.0035029, -0.0011220, -0.0013853, 0.0015135
9: -0.0004789, 0.0020265, -0.0005628, 0.0021979, -0.0017549, 0.0016063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013663
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013770
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9893543, 0.9931653, 0.9891866, 0.9933376, -0.0025692, 0.0026081
1: -0.0039166, -0.0029670, -0.0039584, -0.0029240, -0.0006402, 0.0006499
2: 0.0056695, 0.0107018, 0.0054419, 0.0109232, -0.0034440, 0.0033926
3: -0.0061441, -0.0038536, -0.0062449, -0.0037500, -0.0015441, 0.0015676
4: 0.0016252, 0.0025992, 0.0015812, 0.0026421, -0.0006666, 0.0006566
5: 0.0060901, 0.0124195, 0.0058039, 0.0126980, -0.0043317, 0.0042670
6: -0.0016114, -0.0000049, -0.0016821, 0.0000677, -0.0010830, 0.0010994
7: -0.0073067, -0.0031503, -0.0074896, -0.0029624, -0.0028020, 0.0028445
8: -0.0034067, -0.0012209, -0.0035029, -0.0011220, -0.0014736, 0.0014959
9: -0.0004482, 0.0020864, -0.0005628, 0.0021979, -0.0017346, 0.0017087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013664
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013778
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9892616, 0.9932584, -0.0024226, 0.0026061
1: -0.0039097, -0.0029337, -0.0039397, -0.0029438, -0.0006036, 0.0006494
2: 0.0054932, 0.0106653, 0.0055465, 0.0108243, -0.0034414, 0.0031990
3: -0.0061275, -0.0037734, -0.0061999, -0.0037977, -0.0014560, 0.0015664
4: 0.0015911, 0.0025921, 0.0016014, 0.0026229, -0.0006661, 0.0006192
5: 0.0058684, 0.0123736, 0.0059355, 0.0125735, -0.0043284, 0.0040235
6: -0.0015997, 0.0000514, -0.0016505, 0.0000343, -0.0010212, 0.0010986
7: -0.0072766, -0.0030047, -0.0074079, -0.0030488, -0.0026422, 0.0028424
8: -0.0033908, -0.0011443, -0.0034599, -0.0011675, -0.0013895, 0.0014948
9: -0.0005370, 0.0020680, -0.0005101, 0.0021480, -0.0017333, 0.0016112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013891
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013989
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9892952, 0.9932564, 0.9892616, 0.9932584, -0.0025463, 0.0025789
1: -0.0039313, -0.0029443, -0.0039397, -0.0029438, -0.0006345, 0.0006426
2: 0.0055491, 0.0107798, 0.0055465, 0.0108243, -0.0034054, 0.0033623
3: -0.0061796, -0.0037988, -0.0061999, -0.0037977, -0.0015304, 0.0015500
4: 0.0016019, 0.0026143, 0.0016014, 0.0026229, -0.0006591, 0.0006508
5: 0.0059388, 0.0125176, 0.0059355, 0.0125735, -0.0042831, 0.0042289
6: -0.0016363, 0.0000335, -0.0016505, 0.0000343, -0.0010734, 0.0010871
7: -0.0073711, -0.0030509, -0.0074079, -0.0030488, -0.0027771, 0.0028127
8: -0.0034405, -0.0011686, -0.0034599, -0.0011675, -0.0014604, 0.0014792
9: -0.0005088, 0.0021256, -0.0005101, 0.0021480, -0.0017152, 0.0016935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013897
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013995
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9891657, 0.9933388, -0.0024732, 0.0026781
1: -0.0039097, -0.0029337, -0.0039635, -0.0029238, -0.0006163, 0.0006673
2: 0.0054932, 0.0106653, 0.0054404, 0.0109507, -0.0035364, 0.0032659
3: -0.0061275, -0.0037734, -0.0062574, -0.0037494, -0.0014865, 0.0016096
4: 0.0015911, 0.0025921, 0.0015809, 0.0026474, -0.0006845, 0.0006321
5: 0.0058684, 0.0123736, 0.0058020, 0.0127326, -0.0044478, 0.0041076
6: -0.0015997, 0.0000514, -0.0016908, 0.0000682, -0.0010426, 0.0011289
7: -0.0072766, -0.0030047, -0.0075123, -0.0029611, -0.0026974, 0.0029208
8: -0.0033908, -0.0011443, -0.0035148, -0.0011214, -0.0014185, 0.0015360
9: -0.0005370, 0.0020680, -0.0005636, 0.0022117, -0.0017811, 0.0016449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013663
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013770
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9892952, 0.9932564, 0.9891657, 0.9933388, -0.0026065, 0.0026496
1: -0.0039313, -0.0029443, -0.0039635, -0.0029238, -0.0006495, 0.0006602
2: 0.0055491, 0.0107798, 0.0054404, 0.0109507, -0.0034988, 0.0034419
3: -0.0061796, -0.0037988, -0.0062574, -0.0037494, -0.0015666, 0.0015925
4: 0.0016019, 0.0026143, 0.0015809, 0.0026474, -0.0006772, 0.0006662
5: 0.0059388, 0.0125176, 0.0058020, 0.0127326, -0.0044005, 0.0043290
6: -0.0016363, 0.0000335, -0.0016908, 0.0000682, -0.0010987, 0.0011169
7: -0.0073711, -0.0030509, -0.0075123, -0.0029611, -0.0028428, 0.0028897
8: -0.0034405, -0.0011686, -0.0035148, -0.0011214, -0.0014950, 0.0015197
9: -0.0005088, 0.0021256, -0.0005636, 0.0022117, -0.0017622, 0.0017335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013664
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013779
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9893686, 0.9932994, -0.0026966, 0.0023414
1: -0.0039670, -0.0029866, -0.0039130, -0.0029336, -0.0006719, 0.0005834
2: 0.0057734, 0.0109691, 0.0054924, 0.0106830, -0.0030917, 0.0035609
3: -0.0062658, -0.0039009, -0.0061355, -0.0037730, -0.0016208, 0.0014072
4: 0.0016453, 0.0026509, 0.0015909, 0.0025956, -0.0005984, 0.0006892
5: 0.0062209, 0.0127557, 0.0058674, 0.0123958, -0.0038886, 0.0044786
6: -0.0016967, -0.0000381, -0.0016054, 0.0000516, -0.0011367, 0.0009870
7: -0.0075275, -0.0032362, -0.0072912, -0.0030041, -0.0029410, 0.0025536
8: -0.0035228, -0.0012660, -0.0033985, -0.0011440, -0.0015467, 0.0013429
9: -0.0003958, 0.0022210, -0.0005374, 0.0020769, -0.0015572, 0.0017934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013144, upper bound: 0.0013907
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013144, upper bound: 0.0013993
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9893485, 0.9933001, -0.0027676, 0.0023906
1: -0.0039827, -0.0029623, -0.0039180, -0.0029334, -0.0006896, 0.0005957
2: 0.0056447, 0.0110525, 0.0054913, 0.0107095, -0.0031568, 0.0036546
3: -0.0063037, -0.0038424, -0.0061476, -0.0037725, -0.0016634, 0.0014368
4: 0.0016204, 0.0026671, 0.0015907, 0.0026007, -0.0006110, 0.0007073
5: 0.0060590, 0.0128605, 0.0058661, 0.0124292, -0.0039704, 0.0045966
6: -0.0017233, 0.0000030, -0.0016138, 0.0000520, -0.0011667, 0.0010077
7: -0.0075964, -0.0031299, -0.0073131, -0.0030032, -0.0030185, 0.0026073
8: -0.0035590, -0.0012101, -0.0034100, -0.0011435, -0.0015874, 0.0013712
9: -0.0004606, 0.0022630, -0.0005379, 0.0020902, -0.0015899, 0.0018407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013879, upper bound: 0.0013907
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013879, upper bound: 0.0013993
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9892759, 0.9933755, -0.0027616, 0.0024105
1: -0.0039670, -0.0029866, -0.0039361, -0.0029146, -0.0006881, 0.0006006
2: 0.0057734, 0.0109691, 0.0053919, 0.0108053, -0.0031831, 0.0036467
3: -0.0062658, -0.0039009, -0.0061912, -0.0037273, -0.0016598, 0.0014488
4: 0.0016453, 0.0026509, 0.0015715, 0.0026192, -0.0006161, 0.0007058
5: 0.0062209, 0.0127557, 0.0057410, 0.0125497, -0.0040035, 0.0045865
6: -0.0016967, -0.0000381, -0.0016444, 0.0000837, -0.0011641, 0.0010161
7: -0.0075275, -0.0032362, -0.0073922, -0.0029211, -0.0030119, 0.0026290
8: -0.0035228, -0.0012660, -0.0034517, -0.0011003, -0.0015839, 0.0013826
9: -0.0003958, 0.0022210, -0.0005880, 0.0021385, -0.0016032, 0.0018367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013545, upper bound: 0.0013692
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013545, upper bound: 0.0013754
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9892550, 0.9933764, -0.0028236, 0.0024628
1: -0.0039827, -0.0029623, -0.0039413, -0.0029144, -0.0007036, 0.0006137
2: 0.0056447, 0.0110525, 0.0053908, 0.0108330, -0.0032521, 0.0037286
3: -0.0063037, -0.0038424, -0.0062038, -0.0037268, -0.0016971, 0.0014802
4: 0.0016204, 0.0026671, 0.0015713, 0.0026246, -0.0006294, 0.0007217
5: 0.0060590, 0.0128605, 0.0057396, 0.0125845, -0.0040903, 0.0046896
6: -0.0017233, 0.0000030, -0.0016533, 0.0000840, -0.0011903, 0.0010382
7: -0.0075964, -0.0031299, -0.0074151, -0.0029202, -0.0030796, 0.0026860
8: -0.0035590, -0.0012101, -0.0034637, -0.0010998, -0.0016195, 0.0014125
9: -0.0004606, 0.0022630, -0.0005885, 0.0021524, -0.0016379, 0.0018779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0013693
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0013754
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9890168, 0.9930874, 0.9893686, 0.9932994, -0.0029070, 0.0023671
1: -0.0040007, -0.0029864, -0.0039130, -0.0029336, -0.0007243, 0.0005898
2: 0.0057722, 0.0111474, 0.0054924, 0.0106830, -0.0031257, 0.0038387
3: -0.0063469, -0.0039004, -0.0061355, -0.0037730, -0.0017472, 0.0014227
4: 0.0016451, 0.0026854, 0.0015909, 0.0025956, -0.0006050, 0.0007430
5: 0.0062194, 0.0129799, 0.0058674, 0.0123958, -0.0039313, 0.0048280
6: -0.0017536, -0.0000377, -0.0016054, 0.0000516, -0.0012254, 0.0009978
7: -0.0076748, -0.0032352, -0.0072912, -0.0030041, -0.0031705, 0.0025816
8: -0.0036002, -0.0012655, -0.0033985, -0.0011440, -0.0016673, 0.0013577
9: -0.0003964, 0.0023108, -0.0005374, 0.0020769, -0.0015743, 0.0019333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0013999
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0014073
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9889566, 0.9931896, 0.9893485, 0.9933001, -0.0029802, 0.0024225
1: -0.0040157, -0.0029609, -0.0039180, -0.0029334, -0.0007426, 0.0006036
2: 0.0056375, 0.0112269, 0.0054913, 0.0107095, -0.0031989, 0.0039354
3: -0.0063831, -0.0038390, -0.0061476, -0.0037725, -0.0017912, 0.0014560
4: 0.0016190, 0.0027008, 0.0015907, 0.0026007, -0.0006191, 0.0007617
5: 0.0060499, 0.0130800, 0.0058661, 0.0124292, -0.0040233, 0.0049497
6: -0.0017790, 0.0000053, -0.0016138, 0.0000520, -0.0012563, 0.0010212
7: -0.0077405, -0.0031239, -0.0073131, -0.0030032, -0.0032504, 0.0026421
8: -0.0036348, -0.0012070, -0.0034100, -0.0011435, -0.0017093, 0.0013894
9: -0.0004643, 0.0023509, -0.0005379, 0.0020902, -0.0016111, 0.0019821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013875, upper bound: 0.0013999
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013875, upper bound: 0.0014073
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9890168, 0.9930874, 0.9892759, 0.9933755, -0.0029720, 0.0024363
1: -0.0040007, -0.0029864, -0.0039361, -0.0029146, -0.0007405, 0.0006071
2: 0.0057722, 0.0111474, 0.0053919, 0.0108053, -0.0032171, 0.0039245
3: -0.0063469, -0.0039004, -0.0061912, -0.0037273, -0.0017862, 0.0014643
4: 0.0016451, 0.0026854, 0.0015715, 0.0026192, -0.0006227, 0.0007596
5: 0.0062194, 0.0129799, 0.0057410, 0.0125497, -0.0040462, 0.0049359
6: -0.0017536, -0.0000377, -0.0016444, 0.0000837, -0.0012528, 0.0010270
7: -0.0076748, -0.0032352, -0.0073922, -0.0029211, -0.0032414, 0.0026571
8: -0.0036002, -0.0012655, -0.0034517, -0.0011003, -0.0017046, 0.0013973
9: -0.0003964, 0.0023108, -0.0005880, 0.0021385, -0.0016203, 0.0019766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013540, upper bound: 0.0013730
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013540, upper bound: 0.0013807
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9889566, 0.9931896, 0.9892550, 0.9933764, -0.0030363, 0.0024947
1: -0.0040157, -0.0029609, -0.0039413, -0.0029144, -0.0007566, 0.0006216
2: 0.0056375, 0.0112269, 0.0053908, 0.0108330, -0.0032942, 0.0040093
3: -0.0063831, -0.0038390, -0.0062038, -0.0037268, -0.0018249, 0.0014994
4: 0.0016190, 0.0027008, 0.0015713, 0.0026246, -0.0006376, 0.0007760
5: 0.0060499, 0.0130800, 0.0057396, 0.0125845, -0.0041432, 0.0050427
6: -0.0017790, 0.0000053, -0.0016533, 0.0000840, -0.0012799, 0.0010516
7: -0.0077405, -0.0031239, -0.0074151, -0.0029202, -0.0033115, 0.0027208
8: -0.0036348, -0.0012070, -0.0034637, -0.0010998, -0.0017415, 0.0014308
9: -0.0004643, 0.0023509, -0.0005885, 0.0021524, -0.0016591, 0.0020193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014267, upper bound: 0.0013730
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014267, upper bound: 0.0013806
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9892815, 0.9932573, -0.0026659, 0.0024588
1: -0.0039670, -0.0029866, -0.0039347, -0.0029441, -0.0006643, 0.0006127
2: 0.0057734, 0.0109691, 0.0055481, 0.0107980, -0.0032468, 0.0035203
3: -0.0062658, -0.0039009, -0.0061879, -0.0037984, -0.0016023, 0.0014778
4: 0.0016453, 0.0026509, 0.0016017, 0.0026178, -0.0006284, 0.0006813
5: 0.0062209, 0.0127557, 0.0059374, 0.0125405, -0.0040836, 0.0044276
6: -0.0016967, -0.0000381, -0.0016421, 0.0000338, -0.0011238, 0.0010365
7: -0.0075275, -0.0032362, -0.0073862, -0.0030501, -0.0029075, 0.0026816
8: -0.0035228, -0.0012660, -0.0034485, -0.0011681, -0.0015290, 0.0014102
9: -0.0003958, 0.0022210, -0.0005093, 0.0021348, -0.0016352, 0.0017730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013905
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013989
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890168, 0.9930874, 0.9892815, 0.9932573, -0.0028105, 0.0024184
1: -0.0040007, -0.0029864, -0.0039347, -0.0029441, -0.0007003, 0.0006026
2: 0.0057722, 0.0111474, 0.0055481, 0.0107980, -0.0031935, 0.0037113
3: -0.0063469, -0.0039004, -0.0061879, -0.0037984, -0.0016892, 0.0014535
4: 0.0016451, 0.0026854, 0.0016017, 0.0026178, -0.0006181, 0.0007183
5: 0.0062194, 0.0129799, 0.0059374, 0.0125405, -0.0040166, 0.0046678
6: -0.0017536, -0.0000377, -0.0016421, 0.0000338, -0.0011847, 0.0010194
7: -0.0076748, -0.0032352, -0.0073862, -0.0030501, -0.0030653, 0.0026376
8: -0.0036002, -0.0012655, -0.0034485, -0.0011681, -0.0016120, 0.0013871
9: -0.0003964, 0.0023108, -0.0005093, 0.0021348, -0.0016084, 0.0018692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013999
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0014073
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9891866, 0.9933376, -0.0027317, 0.0025253
1: -0.0039670, -0.0029866, -0.0039584, -0.0029240, -0.0006807, 0.0006292
2: 0.0057734, 0.0109691, 0.0054419, 0.0109232, -0.0033346, 0.0036071
3: -0.0062658, -0.0039009, -0.0062449, -0.0037500, -0.0016418, 0.0015178
4: 0.0016453, 0.0026509, 0.0015812, 0.0026421, -0.0006454, 0.0006981
5: 0.0062209, 0.0127557, 0.0058039, 0.0126980, -0.0041940, 0.0045368
6: -0.0016967, -0.0000381, -0.0016821, 0.0000677, -0.0011515, 0.0010645
7: -0.0075275, -0.0032362, -0.0074896, -0.0029624, -0.0029793, 0.0027541
8: -0.0035228, -0.0012660, -0.0035029, -0.0011220, -0.0015668, 0.0014484
9: -0.0003958, 0.0022210, -0.0005628, 0.0021979, -0.0016795, 0.0018167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013689
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013748
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9890168, 0.9930874, 0.9891866, 0.9933376, -0.0028774, 0.0024867
1: -0.0040007, -0.0029864, -0.0039584, -0.0029240, -0.0007170, 0.0006196
2: 0.0057722, 0.0111474, 0.0054419, 0.0109232, -0.0032837, 0.0037996
3: -0.0063469, -0.0039004, -0.0062449, -0.0037500, -0.0017294, 0.0014946
4: 0.0016451, 0.0026854, 0.0015812, 0.0026421, -0.0006355, 0.0007354
5: 0.0062194, 0.0129799, 0.0058039, 0.0126980, -0.0041300, 0.0047789
6: -0.0017536, -0.0000377, -0.0016821, 0.0000677, -0.0012129, 0.0010482
7: -0.0076748, -0.0032352, -0.0074896, -0.0029624, -0.0031382, 0.0027121
8: -0.0036002, -0.0012655, -0.0035029, -0.0011220, -0.0016504, 0.0014263
9: -0.0003964, 0.0023108, -0.0005628, 0.0021979, -0.0016538, 0.0019137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013730
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013807
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9892616, 0.9932584, -0.0027306, 0.0025078
1: -0.0039827, -0.0029623, -0.0039397, -0.0029438, -0.0006804, 0.0006249
2: 0.0056447, 0.0110525, 0.0055465, 0.0108243, -0.0033115, 0.0036057
3: -0.0063037, -0.0038424, -0.0061999, -0.0037977, -0.0016412, 0.0015072
4: 0.0016204, 0.0026671, 0.0016014, 0.0026229, -0.0006409, 0.0006979
5: 0.0060590, 0.0128605, 0.0059355, 0.0125735, -0.0041650, 0.0045350
6: -0.0017233, 0.0000030, -0.0016505, 0.0000343, -0.0011510, 0.0010571
7: -0.0075964, -0.0031299, -0.0074079, -0.0030488, -0.0029781, 0.0027351
8: -0.0035590, -0.0012101, -0.0034599, -0.0011675, -0.0015661, 0.0014383
9: -0.0004606, 0.0022630, -0.0005101, 0.0021480, -0.0016678, 0.0018160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013905
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013989
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9889566, 0.9931896, 0.9892616, 0.9932584, -0.0028561, 0.0024746
1: -0.0040157, -0.0029609, -0.0039397, -0.0029438, -0.0007117, 0.0006166
2: 0.0056375, 0.0112269, 0.0055465, 0.0108243, -0.0032676, 0.0037714
3: -0.0063831, -0.0038390, -0.0061999, -0.0037977, -0.0017166, 0.0014873
4: 0.0016190, 0.0027008, 0.0016014, 0.0026229, -0.0006324, 0.0007300
5: 0.0060499, 0.0130800, 0.0059355, 0.0125735, -0.0041098, 0.0047435
6: -0.0017790, 0.0000053, -0.0016505, 0.0000343, -0.0012039, 0.0010431
7: -0.0077405, -0.0031239, -0.0074079, -0.0030488, -0.0031150, 0.0026989
8: -0.0036348, -0.0012070, -0.0034599, -0.0011675, -0.0016381, 0.0014193
9: -0.0004643, 0.0023509, -0.0005101, 0.0021480, -0.0016457, 0.0018995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013999
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0014073
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9891657, 0.9933388, -0.0027812, 0.0025797
1: -0.0039827, -0.0029623, -0.0039635, -0.0029238, -0.0006930, 0.0006428
2: 0.0056447, 0.0110525, 0.0054404, 0.0109507, -0.0034065, 0.0036726
3: -0.0063037, -0.0038424, -0.0062574, -0.0037494, -0.0016716, 0.0015505
4: 0.0016204, 0.0026671, 0.0015809, 0.0026474, -0.0006593, 0.0007108
5: 0.0060590, 0.0128605, 0.0058020, 0.0127326, -0.0042844, 0.0046191
6: -0.0017233, 0.0000030, -0.0016908, 0.0000682, -0.0011724, 0.0010874
7: -0.0075964, -0.0031299, -0.0075123, -0.0029611, -0.0030333, 0.0028135
8: -0.0035590, -0.0012101, -0.0035148, -0.0011214, -0.0015952, 0.0014796
9: -0.0004606, 0.0022630, -0.0005636, 0.0022117, -0.0017157, 0.0018497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013689
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013748
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9889566, 0.9931896, 0.9891657, 0.9933388, -0.0029163, 0.0025452
1: -0.0040157, -0.0029609, -0.0039635, -0.0029238, -0.0007267, 0.0006342
2: 0.0056375, 0.0112269, 0.0054404, 0.0109507, -0.0033609, 0.0038510
3: -0.0063831, -0.0038390, -0.0062574, -0.0037494, -0.0017528, 0.0015298
4: 0.0016190, 0.0027008, 0.0015809, 0.0026474, -0.0006505, 0.0007453
5: 0.0060499, 0.0130800, 0.0058020, 0.0127326, -0.0042272, 0.0048435
6: -0.0017790, 0.0000053, -0.0016908, 0.0000682, -0.0012293, 0.0010729
7: -0.0077405, -0.0031239, -0.0075123, -0.0029611, -0.0031807, 0.0027759
8: -0.0036348, -0.0012070, -0.0035148, -0.0011214, -0.0016727, 0.0014598
9: -0.0004643, 0.0023509, -0.0005636, 0.0022117, -0.0016927, 0.0019396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013730
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013807
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9890770, 0.9931848, -0.0022849, 0.0027714
1: -0.0038941, -0.0029555, -0.0039857, -0.0029621, -0.0005693, 0.0006906
2: 0.0056086, 0.0105828, 0.0056438, 0.0110679, -0.0036596, 0.0030172
3: -0.0060900, -0.0038259, -0.0063108, -0.0038419, -0.0013733, 0.0016657
4: 0.0016134, 0.0025762, 0.0016202, 0.0026701, -0.0007083, 0.0005840
5: 0.0060135, 0.0122699, 0.0060579, 0.0128800, -0.0046028, 0.0037949
6: -0.0015734, 0.0000145, -0.0017282, 0.0000033, -0.0009632, 0.0011682
7: -0.0072085, -0.0031000, -0.0076091, -0.0031291, -0.0024920, 0.0030226
8: -0.0033550, -0.0011944, -0.0035657, -0.0012097, -0.0013105, 0.0015895
9: -0.0004789, 0.0020265, -0.0004611, 0.0022708, -0.0018431, 0.0015196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013158, upper bound: 0.0013782
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013158, upper bound: 0.0013879
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9890558, 0.9931858, -0.0023449, 0.0028082
1: -0.0039097, -0.0029337, -0.0039910, -0.0029619, -0.0005843, 0.0006997
2: 0.0054932, 0.0106653, 0.0056424, 0.0110960, -0.0037083, 0.0030964
3: -0.0061275, -0.0037734, -0.0063236, -0.0038413, -0.0014093, 0.0016878
4: 0.0015911, 0.0025921, 0.0016200, 0.0026755, -0.0007177, 0.0005993
5: 0.0058684, 0.0123736, 0.0060562, 0.0129154, -0.0046640, 0.0038944
6: -0.0015997, 0.0000514, -0.0017372, 0.0000037, -0.0009884, 0.0011838
7: -0.0072766, -0.0030047, -0.0076324, -0.0031280, -0.0025574, 0.0030628
8: -0.0033908, -0.0011443, -0.0035779, -0.0012091, -0.0013449, 0.0016107
9: -0.0005370, 0.0020680, -0.0004618, 0.0022849, -0.0018677, 0.0015595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013993, upper bound: 0.0013782
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013993, upper bound: 0.0013879
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9889449, 0.9931908, -0.0023219, 0.0029814
1: -0.0038941, -0.0029555, -0.0040186, -0.0029606, -0.0005785, 0.0007429
2: 0.0056086, 0.0105828, 0.0056358, 0.0112423, -0.0039369, 0.0030660
3: -0.0060900, -0.0038259, -0.0063901, -0.0038383, -0.0013955, 0.0017919
4: 0.0016134, 0.0025762, 0.0016187, 0.0027038, -0.0007620, 0.0005934
5: 0.0060135, 0.0122699, 0.0060478, 0.0130993, -0.0049516, 0.0038562
6: -0.0015734, 0.0000145, -0.0017839, 0.0000058, -0.0009788, 0.0012568
7: -0.0072085, -0.0031000, -0.0077532, -0.0031225, -0.0025323, 0.0032516
8: -0.0033550, -0.0011944, -0.0036415, -0.0012063, -0.0013317, 0.0017100
9: -0.0004789, 0.0020265, -0.0004651, 0.0023586, -0.0019828, 0.0015442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013236, upper bound: 0.0013779
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013236, upper bound: 0.0013875
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9889243, 0.9931922, -0.0023716, 0.0030164
1: -0.0039097, -0.0029337, -0.0040238, -0.0029603, -0.0005909, 0.0007516
2: 0.0054932, 0.0106653, 0.0056340, 0.0112698, -0.0039831, 0.0031317
3: -0.0061275, -0.0037734, -0.0064026, -0.0038375, -0.0014254, 0.0018129
4: 0.0015911, 0.0025921, 0.0016183, 0.0027091, -0.0007709, 0.0006061
5: 0.0058684, 0.0123736, 0.0060455, 0.0131339, -0.0050097, 0.0039389
6: -0.0015997, 0.0000514, -0.0017927, 0.0000064, -0.0009997, 0.0012715
7: -0.0072766, -0.0030047, -0.0077759, -0.0031210, -0.0025866, 0.0032898
8: -0.0033908, -0.0011443, -0.0036534, -0.0012055, -0.0013603, 0.0017301
9: -0.0005370, 0.0020680, -0.0004661, 0.0023724, -0.0020061, 0.0015773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013779
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013875
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9890770, 0.9931848, -0.0024984, 0.0025622
1: -0.0039670, -0.0029866, -0.0039857, -0.0029621, -0.0006225, 0.0006384
2: 0.0057734, 0.0109691, 0.0056438, 0.0110679, -0.0033833, 0.0032991
3: -0.0062658, -0.0039009, -0.0063108, -0.0038419, -0.0015016, 0.0015399
4: 0.0016453, 0.0026509, 0.0016202, 0.0026701, -0.0006548, 0.0006385
5: 0.0062209, 0.0127557, 0.0060579, 0.0128800, -0.0042553, 0.0041494
6: -0.0016967, -0.0000381, -0.0017282, 0.0000033, -0.0010532, 0.0010800
7: -0.0075275, -0.0032362, -0.0076091, -0.0031291, -0.0027249, 0.0027944
8: -0.0035228, -0.0012660, -0.0035657, -0.0012097, -0.0014330, 0.0014695
9: -0.0003958, 0.0022210, -0.0004611, 0.0022708, -0.0017040, 0.0016616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013907
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013993
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9890558, 0.9931858, -0.0025720, 0.0025947
1: -0.0039827, -0.0029623, -0.0039910, -0.0029619, -0.0006409, 0.0006465
2: 0.0056447, 0.0110525, 0.0056424, 0.0110960, -0.0034263, 0.0033963
3: -0.0063037, -0.0038424, -0.0063236, -0.0038413, -0.0015458, 0.0015595
4: 0.0016204, 0.0026671, 0.0016200, 0.0026755, -0.0006632, 0.0006573
5: 0.0060590, 0.0128605, 0.0060562, 0.0129154, -0.0043094, 0.0042716
6: -0.0017233, 0.0000030, -0.0017372, 0.0000037, -0.0010842, 0.0010938
7: -0.0075964, -0.0031299, -0.0076324, -0.0031280, -0.0028051, 0.0028299
8: -0.0035590, -0.0012101, -0.0035779, -0.0012091, -0.0014752, 0.0014882
9: -0.0004606, 0.0022630, -0.0004618, 0.0022849, -0.0017257, 0.0017105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013869, upper bound: 0.0013907
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013869, upper bound: 0.0013993
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9889449, 0.9931908, -0.0024690, 0.0026773
1: -0.0039670, -0.0029866, -0.0040186, -0.0029606, -0.0006152, 0.0006671
2: 0.0057734, 0.0109691, 0.0056358, 0.0112423, -0.0035353, 0.0032603
3: -0.0062658, -0.0039009, -0.0063901, -0.0038383, -0.0014839, 0.0016091
4: 0.0016453, 0.0026509, 0.0016187, 0.0027038, -0.0006843, 0.0006310
5: 0.0062209, 0.0127557, 0.0060478, 0.0130993, -0.0044465, 0.0041006
6: -0.0016967, -0.0000381, -0.0017839, 0.0000058, -0.0010408, 0.0011286
7: -0.0075275, -0.0032362, -0.0077532, -0.0031225, -0.0026928, 0.0029200
8: -0.0035228, -0.0012660, -0.0036415, -0.0012063, -0.0014161, 0.0015356
9: -0.0003958, 0.0022210, -0.0004651, 0.0023586, -0.0017806, 0.0016421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013090, upper bound: 0.0013904
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013090, upper bound: 0.0013989
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9889243, 0.9931922, -0.0025393, 0.0027091
1: -0.0039827, -0.0029623, -0.0040238, -0.0029603, -0.0006327, 0.0006750
2: 0.0056447, 0.0110525, 0.0056340, 0.0112698, -0.0035773, 0.0033531
3: -0.0063037, -0.0038424, -0.0064026, -0.0038375, -0.0015262, 0.0016282
4: 0.0016204, 0.0026671, 0.0016183, 0.0027091, -0.0006924, 0.0006490
5: 0.0060590, 0.0128605, 0.0060455, 0.0131339, -0.0044993, 0.0042173
6: -0.0017233, 0.0000030, -0.0017927, 0.0000064, -0.0010704, 0.0011420
7: -0.0075964, -0.0031299, -0.0077759, -0.0031210, -0.0027694, 0.0029546
8: -0.0035590, -0.0012101, -0.0036534, -0.0012055, -0.0014564, 0.0015538
9: -0.0004606, 0.0022630, -0.0004661, 0.0023724, -0.0018017, 0.0016888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013838, upper bound: 0.0013905
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013838, upper bound: 0.0013989
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9889760, 0.9932753, -0.0023595, 0.0028579
1: -0.0038941, -0.0029555, -0.0040108, -0.0029396, -0.0005879, 0.0007121
2: 0.0056086, 0.0105828, 0.0055242, 0.0112013, -0.0037738, 0.0031156
3: -0.0060900, -0.0038259, -0.0063715, -0.0037875, -0.0014181, 0.0017177
4: 0.0016134, 0.0025762, 0.0015971, 0.0026959, -0.0007304, 0.0006030
5: 0.0060135, 0.0122699, 0.0059074, 0.0130478, -0.0047465, 0.0039187
6: -0.0015734, 0.0000145, -0.0017708, 0.0000415, -0.0009946, 0.0012047
7: -0.0072085, -0.0031000, -0.0077193, -0.0030303, -0.0025733, 0.0031170
8: -0.0033550, -0.0011944, -0.0036237, -0.0011578, -0.0013533, 0.0016392
9: -0.0004789, 0.0020265, -0.0005214, 0.0023379, -0.0019007, 0.0015692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013575, upper bound: 0.0013553
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013575, upper bound: 0.0013652
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9889544, 0.9932764, -0.0024075, 0.0028963
1: -0.0039097, -0.0029337, -0.0040162, -0.0029393, -0.0005999, 0.0007217
2: 0.0054932, 0.0106653, 0.0055228, 0.0112298, -0.0038246, 0.0031791
3: -0.0061275, -0.0037734, -0.0063844, -0.0037869, -0.0014470, 0.0017408
4: 0.0015911, 0.0025921, 0.0015968, 0.0027014, -0.0007402, 0.0006153
5: 0.0058684, 0.0123736, 0.0059057, 0.0130836, -0.0048103, 0.0039984
6: -0.0015997, 0.0000514, -0.0017799, 0.0000419, -0.0010148, 0.0012209
7: -0.0072766, -0.0030047, -0.0077428, -0.0030292, -0.0026257, 0.0031589
8: -0.0033908, -0.0011443, -0.0036360, -0.0011572, -0.0013808, 0.0016612
9: -0.0005370, 0.0020680, -0.0005220, 0.0023523, -0.0019263, 0.0016011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014406, upper bound: 0.0013553
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014406, upper bound: 0.0013652
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9894444, 0.9932113, 0.9888487, 0.9932772, -0.0023794, 0.0030474
1: -0.0038941, -0.0029555, -0.0040425, -0.0029391, -0.0005929, 0.0007593
2: 0.0056086, 0.0105828, 0.0055216, 0.0113694, -0.0040240, 0.0031419
3: -0.0060900, -0.0038259, -0.0064480, -0.0037863, -0.0014301, 0.0018316
4: 0.0016134, 0.0025762, 0.0015966, 0.0027284, -0.0007788, 0.0006081
5: 0.0060135, 0.0122699, 0.0059042, 0.0132591, -0.0050612, 0.0039517
6: -0.0015734, 0.0000145, -0.0018245, 0.0000423, -0.0010030, 0.0012846
7: -0.0072085, -0.0031000, -0.0078581, -0.0030282, -0.0025951, 0.0033236
8: -0.0033550, -0.0011944, -0.0036966, -0.0011567, -0.0013647, 0.0017479
9: -0.0004789, 0.0020265, -0.0005226, 0.0024226, -0.0020267, 0.0015825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013648, upper bound: 0.0013550
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013648, upper bound: 0.0013648
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9893819, 0.9932987, 0.9888287, 0.9932786, -0.0024301, 0.0030870
1: -0.0039097, -0.0029337, -0.0040475, -0.0029387, -0.0006055, 0.0007692
2: 0.0054932, 0.0106653, 0.0055198, 0.0113958, -0.0040764, 0.0032089
3: -0.0061275, -0.0037734, -0.0064600, -0.0037855, -0.0014605, 0.0018554
4: 0.0015911, 0.0025921, 0.0015962, 0.0027335, -0.0007890, 0.0006211
5: 0.0058684, 0.0123736, 0.0059019, 0.0132924, -0.0051270, 0.0040359
6: -0.0015997, 0.0000514, -0.0018329, 0.0000429, -0.0010244, 0.0013013
7: -0.0072766, -0.0030047, -0.0078799, -0.0030267, -0.0026503, 0.0033669
8: -0.0033908, -0.0011443, -0.0037081, -0.0011559, -0.0013938, 0.0017706
9: -0.0005370, 0.0020680, -0.0005236, 0.0024359, -0.0020531, 0.0016162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0013549
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0013648
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9889760, 0.9932753, -0.0025640, 0.0026297
1: -0.0039670, -0.0029866, -0.0040108, -0.0029396, -0.0006389, 0.0006552
2: 0.0057734, 0.0109691, 0.0055242, 0.0112013, -0.0034725, 0.0033857
3: -0.0062658, -0.0039009, -0.0063715, -0.0037875, -0.0015410, 0.0015805
4: 0.0016453, 0.0026509, 0.0015971, 0.0026959, -0.0006721, 0.0006553
5: 0.0062209, 0.0127557, 0.0059074, 0.0130478, -0.0043674, 0.0042583
6: -0.0016967, -0.0000381, -0.0017708, 0.0000415, -0.0010808, 0.0011085
7: -0.0075275, -0.0032362, -0.0077193, -0.0030303, -0.0027964, 0.0028680
8: -0.0035228, -0.0012660, -0.0036237, -0.0011578, -0.0014706, 0.0015083
9: -0.0003958, 0.0022210, -0.0005214, 0.0023379, -0.0017489, 0.0017052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0013693
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0013754
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9889544, 0.9932764, -0.0026288, 0.0026649
1: -0.0039827, -0.0029623, -0.0040162, -0.0029393, -0.0006550, 0.0006640
2: 0.0056447, 0.0110525, 0.0055228, 0.0112298, -0.0035190, 0.0034714
3: -0.0063037, -0.0038424, -0.0063844, -0.0037869, -0.0015800, 0.0016017
4: 0.0016204, 0.0026671, 0.0015968, 0.0027014, -0.0006811, 0.0006719
5: 0.0060590, 0.0128605, 0.0059057, 0.0130836, -0.0044260, 0.0043661
6: -0.0017233, 0.0000030, -0.0017799, 0.0000419, -0.0011082, 0.0011234
7: -0.0075964, -0.0031299, -0.0077428, -0.0030292, -0.0028671, 0.0029065
8: -0.0035590, -0.0012101, -0.0036360, -0.0011572, -0.0015078, 0.0015285
9: -0.0004606, 0.0022630, -0.0005220, 0.0023523, -0.0017724, 0.0017484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014263, upper bound: 0.0013692
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014263, upper bound: 0.0013754
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9891519, 0.9930865, 0.9888487, 0.9932772, -0.0025339, 0.0027421
1: -0.0039670, -0.0029866, -0.0040425, -0.0029391, -0.0006314, 0.0006832
2: 0.0057734, 0.0109691, 0.0055216, 0.0113694, -0.0036209, 0.0033460
3: -0.0062658, -0.0039009, -0.0064480, -0.0037863, -0.0015230, 0.0016481
4: 0.0016453, 0.0026509, 0.0015966, 0.0027284, -0.0007008, 0.0006476
5: 0.0062209, 0.0127557, 0.0059042, 0.0132591, -0.0045541, 0.0042084
6: -0.0016967, -0.0000381, -0.0018245, 0.0000423, -0.0010681, 0.0011559
7: -0.0075275, -0.0032362, -0.0078581, -0.0030282, -0.0027636, 0.0029906
8: -0.0035228, -0.0012660, -0.0036966, -0.0011567, -0.0014534, 0.0015727
9: -0.0003958, 0.0022210, -0.0005226, 0.0024226, -0.0018237, 0.0016852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013486, upper bound: 0.0013689
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013486, upper bound: 0.0013748
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9890887, 0.9931840, 0.9888287, 0.9932786, -0.0025863, 0.0027791
1: -0.0039827, -0.0029623, -0.0040475, -0.0029387, -0.0006444, 0.0006925
2: 0.0056447, 0.0110525, 0.0055198, 0.0113958, -0.0036698, 0.0034152
3: -0.0063037, -0.0038424, -0.0064600, -0.0037855, -0.0015544, 0.0016703
4: 0.0016204, 0.0026671, 0.0015962, 0.0027335, -0.0007103, 0.0006610
5: 0.0060590, 0.0128605, 0.0059019, 0.0132924, -0.0046157, 0.0042954
6: -0.0017233, 0.0000030, -0.0018329, 0.0000429, -0.0010902, 0.0011715
7: -0.0075964, -0.0031299, -0.0078799, -0.0030267, -0.0028207, 0.0030310
8: -0.0035590, -0.0012101, -0.0037081, -0.0011559, -0.0014834, 0.0015940
9: -0.0004606, 0.0022630, -0.0005236, 0.0024359, -0.0018483, 0.0017201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013689
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013748
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893434, 0.9932532, 0.9891601, 0.9931162, -0.0024799, 0.0028025
1: -0.0039193, -0.0029451, -0.0039650, -0.0029792, -0.0006179, 0.0006983
2: 0.0055533, 0.0107161, 0.0057342, 0.0109584, -0.0037007, 0.0032747
3: -0.0061506, -0.0038008, -0.0062609, -0.0038831, -0.0014905, 0.0016844
4: 0.0016027, 0.0026020, 0.0016377, 0.0026489, -0.0007163, 0.0006338
5: 0.0059441, 0.0124375, 0.0061716, 0.0127422, -0.0046545, 0.0041188
6: -0.0016160, 0.0000322, -0.0016933, -0.0000256, -0.0010454, 0.0011814
7: -0.0073186, -0.0030544, -0.0075186, -0.0032038, -0.0027047, 0.0030565
8: -0.0034129, -0.0011704, -0.0035181, -0.0012490, -0.0014224, 0.0016074
9: -0.0005067, 0.0020936, -0.0004156, 0.0022156, -0.0018639, 0.0016493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013106
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013106
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9893507, 0.9933684, 0.9891756, 0.9931158, -0.0024974, 0.0029092
1: -0.0039175, -0.0029164, -0.0039611, -0.0029793, -0.0006223, 0.0007249
2: 0.0054013, 0.0107065, 0.0057349, 0.0109378, -0.0038416, 0.0032978
3: -0.0061462, -0.0037315, -0.0062515, -0.0038834, -0.0015010, 0.0017485
4: 0.0015733, 0.0026001, 0.0016379, 0.0026449, -0.0007435, 0.0006383
5: 0.0057528, 0.0124254, 0.0061724, 0.0127163, -0.0048317, 0.0041478
6: -0.0016129, 0.0000807, -0.0016867, -0.0000258, -0.0010527, 0.0012263
7: -0.0073106, -0.0029288, -0.0075017, -0.0032044, -0.0027238, 0.0031729
8: -0.0034087, -0.0011044, -0.0035092, -0.0012493, -0.0014324, 0.0016686
9: -0.0005833, 0.0020887, -0.0004152, 0.0022052, -0.0019348, 0.0016609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013106
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013106
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9893237, 0.9932544, 0.9890957, 0.9932145, -0.0025308, 0.0028339
1: -0.0039242, -0.0029448, -0.0039810, -0.0029547, -0.0006306, 0.0007061
2: 0.0055517, 0.0107422, 0.0056045, 0.0110433, -0.0037421, 0.0033419
3: -0.0061625, -0.0038000, -0.0062995, -0.0038240, -0.0015211, 0.0017033
4: 0.0016024, 0.0026070, 0.0016126, 0.0026653, -0.0007243, 0.0006468
5: 0.0059421, 0.0124703, 0.0060084, 0.0128490, -0.0047066, 0.0042033
6: -0.0016243, 0.0000327, -0.0017204, 0.0000158, -0.0010668, 0.0011946
7: -0.0073401, -0.0030531, -0.0075888, -0.0030966, -0.0027602, 0.0030908
8: -0.0034242, -0.0011697, -0.0035550, -0.0011926, -0.0014516, 0.0016254
9: -0.0005075, 0.0021067, -0.0004809, 0.0022583, -0.0018847, 0.0016832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013838
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013839
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9893312, 0.9933698, 0.9891104, 0.9932141, -0.0025479, 0.0029310
1: -0.0039223, -0.0029160, -0.0039773, -0.0029548, -0.0006349, 0.0007303
2: 0.0053994, 0.0107324, 0.0056051, 0.0110239, -0.0038703, 0.0033645
3: -0.0061580, -0.0037307, -0.0062907, -0.0038243, -0.0015314, 0.0017616
4: 0.0015729, 0.0026051, 0.0016127, 0.0026615, -0.0007491, 0.0006512
5: 0.0057505, 0.0124580, 0.0060092, 0.0128246, -0.0048679, 0.0042317
6: -0.0016211, 0.0000813, -0.0017142, 0.0000156, -0.0010740, 0.0012355
7: -0.0073320, -0.0029273, -0.0075727, -0.0030972, -0.0027789, 0.0031967
8: -0.0034200, -0.0011036, -0.0035466, -0.0011929, -0.0014614, 0.0016811
9: -0.0005842, 0.0021018, -0.0004806, 0.0022486, -0.0019493, 0.0016945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013838
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013839
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9890053, 0.9931862, 0.9891601, 0.9931162, -0.0026529, 0.0026556
1: -0.0040035, -0.0029618, -0.0039650, -0.0029792, -0.0006610, 0.0006617
2: 0.0056418, 0.0111627, 0.0057342, 0.0109584, -0.0035066, 0.0035031
3: -0.0063539, -0.0038410, -0.0062609, -0.0038831, -0.0015944, 0.0015961
4: 0.0016198, 0.0026884, 0.0016377, 0.0026489, -0.0006787, 0.0006780
5: 0.0060553, 0.0129992, 0.0061716, 0.0127422, -0.0044104, 0.0044059
6: -0.0017585, 0.0000039, -0.0016933, -0.0000256, -0.0011183, 0.0011194
7: -0.0076874, -0.0031275, -0.0075186, -0.0032038, -0.0028933, 0.0028963
8: -0.0036069, -0.0012089, -0.0035181, -0.0012490, -0.0015216, 0.0015231
9: -0.0004621, 0.0023185, -0.0004156, 0.0022156, -0.0017661, 0.0017643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0013236
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0013236
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9890153, 0.9932990, 0.9891756, 0.9931158, -0.0026695, 0.0027685
1: -0.0040011, -0.0029337, -0.0039611, -0.0029793, -0.0006652, 0.0006898
2: 0.0054929, 0.0111495, 0.0057349, 0.0109378, -0.0036558, 0.0035250
3: -0.0063479, -0.0037733, -0.0062515, -0.0038834, -0.0016044, 0.0016639
4: 0.0015910, 0.0026859, 0.0016379, 0.0026449, -0.0007076, 0.0006823
5: 0.0058681, 0.0129826, 0.0061724, 0.0127163, -0.0045980, 0.0044335
6: -0.0017543, 0.0000514, -0.0016867, -0.0000258, -0.0011253, 0.0011670
7: -0.0076765, -0.0030045, -0.0075017, -0.0032044, -0.0029114, 0.0030194
8: -0.0036011, -0.0011442, -0.0035092, -0.0012493, -0.0015311, 0.0015879
9: -0.0005371, 0.0023119, -0.0004152, 0.0022052, -0.0018412, 0.0017754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013832, upper bound: 0.0013236
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013832, upper bound: 0.0013236
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9889851, 0.9931878, 0.9890957, 0.9932145, -0.0026789, 0.0026839
1: -0.0040086, -0.0029614, -0.0039810, -0.0029547, -0.0006675, 0.0006687
2: 0.0056398, 0.0111893, 0.0056045, 0.0110433, -0.0035440, 0.0035374
3: -0.0063660, -0.0038401, -0.0062995, -0.0038240, -0.0016101, 0.0016131
4: 0.0016195, 0.0026936, 0.0016126, 0.0026653, -0.0006859, 0.0006847
5: 0.0060528, 0.0130327, 0.0060084, 0.0128490, -0.0044574, 0.0044491
6: -0.0017670, 0.0000046, -0.0017204, 0.0000158, -0.0011292, 0.0011313
7: -0.0077094, -0.0031258, -0.0075888, -0.0030966, -0.0029217, 0.0029271
8: -0.0036184, -0.0012080, -0.0035550, -0.0011926, -0.0015365, 0.0015394
9: -0.0004631, 0.0023319, -0.0004809, 0.0022583, -0.0017850, 0.0017816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0014073
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0014073
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9889933, 0.9933006, 0.9891104, 0.9932141, -0.0026956, 0.0027816
1: -0.0040065, -0.0029333, -0.0039773, -0.0029548, -0.0006717, 0.0006931
2: 0.0054908, 0.0111784, 0.0056051, 0.0110239, -0.0036731, 0.0035594
3: -0.0063610, -0.0037723, -0.0062907, -0.0038243, -0.0016201, 0.0016718
4: 0.0015906, 0.0026914, 0.0016127, 0.0026615, -0.0007109, 0.0006889
5: 0.0058655, 0.0130190, 0.0060092, 0.0128246, -0.0046198, 0.0044768
6: -0.0017635, 0.0000521, -0.0017142, 0.0000156, -0.0011363, 0.0011726
7: -0.0077004, -0.0030028, -0.0075727, -0.0030972, -0.0029399, 0.0030338
8: -0.0036137, -0.0011433, -0.0035466, -0.0011929, -0.0015461, 0.0015954
9: -0.0005382, 0.0023264, -0.0004806, 0.0022486, -0.0018500, 0.0017927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013831, upper bound: 0.0014073
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013831, upper bound: 0.0014073
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9893434, 0.9932532, 0.9890635, 0.9932038, -0.0025366, 0.0028826
1: -0.0039193, -0.0029451, -0.0039890, -0.0029574, -0.0006321, 0.0007183
2: 0.0055533, 0.0107161, 0.0056187, 0.0110858, -0.0038064, 0.0033496
3: -0.0061506, -0.0038008, -0.0063189, -0.0038305, -0.0015246, 0.0017325
4: 0.0016027, 0.0026020, 0.0016154, 0.0026735, -0.0007367, 0.0006483
5: 0.0059441, 0.0124375, 0.0060263, 0.0129025, -0.0047875, 0.0042129
6: -0.0016160, 0.0000322, -0.0017340, 0.0000113, -0.0010693, 0.0012151
7: -0.0073186, -0.0030544, -0.0076239, -0.0031084, -0.0027665, 0.0031439
8: -0.0034129, -0.0011704, -0.0035735, -0.0011988, -0.0014549, 0.0016533
9: -0.0005067, 0.0020936, -0.0004737, 0.0022798, -0.0019171, 0.0016870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014288, upper bound: 0.0012972
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014288, upper bound: 0.0012972
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9893507, 0.9933684, 0.9890794, 0.9932032, -0.0025541, 0.0029935
1: -0.0039175, -0.0029164, -0.0039851, -0.0029575, -0.0006364, 0.0007459
2: 0.0054013, 0.0107065, 0.0056194, 0.0110648, -0.0039529, 0.0033726
3: -0.0061462, -0.0037315, -0.0063093, -0.0038308, -0.0015351, 0.0017992
4: 0.0015733, 0.0026001, 0.0016155, 0.0026695, -0.0007651, 0.0006528
5: 0.0057528, 0.0124254, 0.0060272, 0.0128761, -0.0049718, 0.0042419
6: -0.0016129, 0.0000807, -0.0017273, 0.0000111, -0.0010766, 0.0012619
7: -0.0073106, -0.0029288, -0.0076066, -0.0031090, -0.0027856, 0.0032649
8: -0.0034087, -0.0011044, -0.0035644, -0.0011991, -0.0014649, 0.0017170
9: -0.0005833, 0.0020887, -0.0004734, 0.0022692, -0.0019909, 0.0016986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 138

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0012972
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0012972
time: 0.91 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.36 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013287, upper bound: 0.0013951
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013287, upper bound: 0.0014052
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014051, upper bound: 0.0013951
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014051, upper bound: 0.0014051
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013690, upper bound: 0.0013733
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013691, upper bound: 0.0013830
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014443, upper bound: 0.0013733
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014443, upper bound: 0.0013830
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0013897
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013255, upper bound: 0.0013995
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014017, upper bound: 0.0013897
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014017, upper bound: 0.0013995
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013663, upper bound: 0.0013663
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013663, upper bound: 0.0013778
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014410, upper bound: 0.0013663
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014410, upper bound: 0.0013779
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013891
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013989
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013897
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013219, upper bound: 0.0013995
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013663
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013770
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013664
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013614, upper bound: 0.0013778
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013891
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013989
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013897
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013995
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013663
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013770
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013664
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0013779
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013144, upper bound: 0.0013907
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013144, upper bound: 0.0013993
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013879, upper bound: 0.0013907
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013879, upper bound: 0.0013993
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013545, upper bound: 0.0013692
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013545, upper bound: 0.0013754
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0013693
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014273, upper bound: 0.0013754
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0013999
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0014073
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013875, upper bound: 0.0013999
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013875, upper bound: 0.0014073
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013540, upper bound: 0.0013730
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013540, upper bound: 0.0013807
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014267, upper bound: 0.0013730
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014267, upper bound: 0.0013806
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013905
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013989
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0013999
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013096, upper bound: 0.0014073
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013689
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013748
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013730
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013481, upper bound: 0.0013807
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013905
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013989
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0013999
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013835, upper bound: 0.0014073
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013689
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013748
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013730
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014225, upper bound: 0.0013807
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013158, upper bound: 0.0013782
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013158, upper bound: 0.0013879
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013993, upper bound: 0.0013782
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013993, upper bound: 0.0013879
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013236, upper bound: 0.0013779
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013236, upper bound: 0.0013875
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013779
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014073, upper bound: 0.0013875
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013907
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0013993
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013869, upper bound: 0.0013907
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013869, upper bound: 0.0013993
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013090, upper bound: 0.0013904
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013090, upper bound: 0.0013989
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013838, upper bound: 0.0013905
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013838, upper bound: 0.0013989
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013575, upper bound: 0.0013553
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013575, upper bound: 0.0013652
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014406, upper bound: 0.0013553
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014406, upper bound: 0.0013652
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013648, upper bound: 0.0013550
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013648, upper bound: 0.0013648
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0013549
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0013648
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0013693
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013491, upper bound: 0.0013754
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014263, upper bound: 0.0013692
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014263, upper bound: 0.0013754
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013486, upper bound: 0.0013689
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013486, upper bound: 0.0013748
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013689
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013748
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013106
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013106
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013106
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013106
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013838
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013904, upper bound: 0.0013839
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013838
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013989, upper bound: 0.0013839
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0013236
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0013236
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013832, upper bound: 0.0013236
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013832, upper bound: 0.0013236
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0014073
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013744, upper bound: 0.0014073
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013831, upper bound: 0.0014073
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0013831, upper bound: 0.0014073
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014288, upper bound: 0.0012972
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014288, upper bound: 0.0012972
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0012972
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.36
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0012972
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014398, upper bound: 0.0013618
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014474, upper bound: 0.0013618
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014129, upper bound: 0.0013070
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013070
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014129, upper bound: 0.0013807
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014226, upper bound: 0.0013807
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013529, upper bound: 0.0014828
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014219, upper bound: 0.0014827
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013529, upper bound: 0.0014828
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014219, upper bound: 0.0014828
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013471, upper bound: 0.0014766
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0014766
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013471, upper bound: 0.0014766
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014162, upper bound: 0.0014766
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014756
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014766
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014755
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013462, upper bound: 0.0014766
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014756
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014766
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014756
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014134, upper bound: 0.0014766
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013406, upper bound: 0.0014857
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014065, upper bound: 0.0014857
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013406, upper bound: 0.0014857
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014065, upper bound: 0.0014857
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014911
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014047, upper bound: 0.0014911
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013385, upper bound: 0.0014911
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014047, upper bound: 0.0014911
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014813
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014912
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014813
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013367, upper bound: 0.0014911
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014813
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014911
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014813
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014007, upper bound: 0.0014912
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013429, upper bound: 0.0014682
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014188, upper bound: 0.0014682
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013429, upper bound: 0.0014682
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014188, upper bound: 0.0014682
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013357, upper bound: 0.0014857
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014063, upper bound: 0.0014857
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013357, upper bound: 0.0014857
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014063, upper bound: 0.0014857
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014661
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014218, upper bound: 0.0014661
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013510, upper bound: 0.0014660
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014218, upper bound: 0.0014661
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0014814
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014814
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013378, upper bound: 0.0014813
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0014011, upper bound: 0.0014814
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0013495
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013807, upper bound: 0.0013495
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0014227
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013807, upper bound: 0.0014226
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0013648
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0013648
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0014474
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0014474
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0013495
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013806, upper bound: 0.0013495
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013730, upper bound: 0.0014226
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013806, upper bound: 0.0014226
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0013648
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0013648
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013507, upper bound: 0.0014474
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 0, lower bound: -0.0013618, upper bound: 0.0014474

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.95 + 597.98 = 600.93 seconds
