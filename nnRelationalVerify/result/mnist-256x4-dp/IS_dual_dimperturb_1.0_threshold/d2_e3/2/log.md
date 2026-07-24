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
Threshold: 0.0157495625


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293)
1: (-0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382)
2: (0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210)
3: (-0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023)
4: (0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575)
5: (0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635)
6: (-0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247)
7: (-0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719)
8: (-0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123)
9: (-0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 2.25 = 3.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0179995, upper bound: 0.0179995

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0177482, upper bound: 0.0175763
time: 2.11 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0176501, upper bound: 0.0176501
time: 1.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.93
Output dim: 0, lower bound: -0.0177482, upper bound: 0.0175763
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.93
Output dim: 0, lower bound: -0.0176501, upper bound: 0.0176501

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9737979, 0.9958722, 0.9732662, 0.9961327, -0.0223348, 0.0226060
1: -0.0046333, -0.0022925, -0.0046410, -0.0022276, -0.0024057, 0.0023485
2: 0.0020951, 0.0145001, 0.0017510, 0.0145411, -0.0124460, 0.0127491
3: -0.0080831, -0.0022267, -0.0081101, -0.0020701, -0.0060130, 0.0058834
4: 0.0009334, 0.0042434, 0.0008668, 0.0042873, -0.0033539, 0.0033766
5: 0.0015945, 0.0266156, 0.0011618, 0.0270390, -0.0254446, 0.0254538
6: -0.0028239, 0.0011361, -0.0028370, 0.0012460, -0.0040698, 0.0039731
7: -0.0104439, -0.0001981, -0.0104777, 0.0000860, -0.0105299, 0.0102796
8: -0.0050565, 0.0003317, -0.0050743, 0.0004811, -0.0055376, 0.0054060
9: -0.0022484, 0.0039994, -0.0024217, 0.0040200, -0.0062685, 0.0064211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159298, upper bound: 0.0162511
time: 1.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158207, upper bound: 0.0156516
time: 1.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9693661, 0.9959902, 0.9732611, 0.9961358, -0.0267697, 0.0227291
1: -0.0046977, -0.0022631, -0.0046411, -0.0022268, -0.0024709, 0.0023780
2: 0.0019392, 0.0148415, 0.0017468, 0.0145415, -0.0126023, 0.0130947
3: -0.0083077, -0.0021558, -0.0081103, -0.0020682, -0.0062395, 0.0059546
4: 0.0009032, 0.0046087, 0.0008660, 0.0042877, -0.0033845, 0.0037427
5: 0.0013984, 0.0301449, 0.0011565, 0.0270431, -0.0256447, 0.0289884
6: -0.0029329, 0.0011859, -0.0028371, 0.0012473, -0.0041802, 0.0040230
7: -0.0107259, -0.0000694, -0.0104781, 0.0000895, -0.0108154, 0.0104087
8: -0.0052048, 0.0011582, -0.0050745, 0.0004829, -0.0056877, 0.0062327
9: -0.0023270, 0.0041714, -0.0024238, 0.0040202, -0.0063472, 0.0065952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158291, upper bound: 0.0162939
time: 1.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0157124, upper bound: 0.0157124
time: 1.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.96 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0159298, upper bound: 0.0162511
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0158207, upper bound: 0.0156516
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0158291, upper bound: 0.0162939
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 3.96
Output dim: 0, lower bound: -0.0157124, upper bound: 0.0157124

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9737979, 0.9958722, 0.9735343, 0.9959509, -0.0221530, 0.0223379
1: -0.0046333, -0.0022925, -0.0046371, -0.0022729, -0.0023604, 0.0023446
2: 0.0020951, 0.0145001, 0.0019911, 0.0145204, -0.0124254, 0.0125090
3: -0.0080831, -0.0022267, -0.0080965, -0.0021794, -0.0059037, 0.0058698
4: 0.0009334, 0.0042434, 0.0009133, 0.0042652, -0.0033318, 0.0033302
5: 0.0015945, 0.0266156, 0.0014637, 0.0268255, -0.0252310, 0.0251519
6: -0.0028239, 0.0011361, -0.0028304, 0.0011693, -0.0039932, 0.0039665
7: -0.0104439, -0.0001981, -0.0104607, -0.0001122, -0.0103317, 0.0102626
8: -0.0050565, 0.0003317, -0.0050653, 0.0003768, -0.0054333, 0.0053970
9: -0.0022484, 0.0039994, -0.0023008, 0.0040096, -0.0062581, 0.0063002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0150015, upper bound: 0.0155161
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157067, upper bound: 0.0160313
time: 1.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9740670, 0.9956561, 0.9440659, 0.9954470, -0.0213800, 0.0515902
1: -0.0046294, -0.0023463, -0.0050656, -0.0019621, -0.0026673, 0.0027192
2: 0.0023803, 0.0144794, 0.0026564, 0.0167908, -0.0144104, 0.0118229
3: -0.0080695, -0.0023566, -0.0095898, -0.0024822, -0.0055873, 0.0072333
4: 0.0009886, 0.0042212, 0.0010420, 0.0066940, -0.0057054, 0.0031792
5: 0.0019533, 0.0264013, 0.0023005, 0.0502935, -0.0483402, 0.0241007
6: -0.0028173, 0.0010451, -0.0035551, 0.0015759, -0.0043932, 0.0046002
7: -0.0104268, -0.0004337, -0.0123359, -0.0006618, -0.0097650, 0.0119021
8: -0.0050475, 0.0002078, -0.0060515, 0.0093065, -0.0143540, 0.0062592
9: -0.0021048, 0.0039890, -0.0019657, 0.0051531, -0.0072579, 0.0059547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0147906, upper bound: 0.0146409
time: 1.28 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155976, upper bound: 0.0154345
time: 1.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9693661, 0.9959902, 0.9735292, 0.9959529, -0.0265868, 0.0224611
1: -0.0046977, -0.0022631, -0.0046372, -0.0022724, -0.0024253, 0.0023741
2: 0.0019392, 0.0148415, 0.0019885, 0.0145208, -0.0125816, 0.0128531
3: -0.0083077, -0.0021558, -0.0080967, -0.0021782, -0.0061295, 0.0059410
4: 0.0009032, 0.0046087, 0.0009128, 0.0042656, -0.0033624, 0.0036960
5: 0.0013984, 0.0301449, 0.0014604, 0.0268296, -0.0254312, 0.0286845
6: -0.0029329, 0.0011859, -0.0028305, 0.0011701, -0.0041030, 0.0040164
7: -0.0107259, -0.0000694, -0.0104610, -0.0001101, -0.0106158, 0.0103916
8: -0.0052048, 0.0011582, -0.0050655, 0.0003780, -0.0055828, 0.0062237
9: -0.0023270, 0.0041714, -0.0023021, 0.0040098, -0.0063368, 0.0064735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0149035, upper bound: 0.0155392
time: 1.30 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0156057, upper bound: 0.0160710
time: 2.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.83 seconds
IS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0150015, upper bound: 0.0155161
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0157067, upper bound: 0.0160313
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0147906, upper bound: 0.0146409
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0155976, upper bound: 0.0154345
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0149035, upper bound: 0.0155392
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -0.0156057, upper bound: 0.0160710

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.9737979, 0.9958722, 0.9737952, 0.9957437, -0.0219458, 0.0220770
1: -0.0046333, -0.0022925, -0.0046333, -0.0023245, -0.0023088, 0.0023408
2: 0.0020951, 0.0145001, 0.0022647, 0.0145003, -0.0124053, 0.0122354
3: -0.0080831, -0.0022267, -0.0080833, -0.0023039, -0.0057792, 0.0058566
4: 0.0009334, 0.0042434, 0.0009662, 0.0042436, -0.0033103, 0.0032772
5: 0.0015945, 0.0266156, 0.0018079, 0.0266177, -0.0250232, 0.0248077
6: -0.0028239, 0.0011361, -0.0028240, 0.0010820, -0.0039059, 0.0039601
7: -0.0104439, -0.0001981, -0.0104441, -0.0003382, -0.0101057, 0.0102460
8: -0.0050565, 0.0003317, -0.0050566, 0.0002580, -0.0053145, 0.0053883
9: -0.0022484, 0.0039994, -0.0021630, 0.0039995, -0.0062479, 0.0061624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0152311, upper bound: 0.0154550
time: 1.74 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0150468, upper bound: 0.0154317
time: 1.64 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.9693661, 0.9959902, 0.9737904, 0.9957446, -0.0263785, 0.0221998
1: -0.0046977, -0.0022631, -0.0046334, -0.0023243, -0.0023734, 0.0023703
2: 0.0019392, 0.0148415, 0.0022635, 0.0145007, -0.0125615, 0.0125780
3: -0.0083077, -0.0021558, -0.0080835, -0.0023034, -0.0060043, 0.0059277
4: 0.0009032, 0.0046087, 0.0009660, 0.0042441, -0.0033408, 0.0036427
5: 0.0013984, 0.0301449, 0.0018064, 0.0266216, -0.0252231, 0.0283385
6: -0.0029329, 0.0011859, -0.0028241, 0.0010823, -0.0040152, 0.0040100
7: -0.0107259, -0.0000694, -0.0104444, -0.0003372, -0.0103887, 0.0103750
8: -0.0052048, 0.0011582, -0.0050567, 0.0002585, -0.0054633, 0.0062149
9: -0.0023270, 0.0041714, -0.0021636, 0.0039997, -0.0063266, 0.0063350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0149726, upper bound: 0.0155622
time: 1.70 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0149692, upper bound: 0.0154751
time: 1.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.71 seconds
IS_A1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 0, lower bound: -0.0152311, upper bound: 0.0154550
IS_A1_B1_B2_A2, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 0, lower bound: -0.0150468, upper bound: 0.0154317
IS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 0, lower bound: -0.0149726, upper bound: 0.0155622
IS_A2_B1_B2_B2, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 0, lower bound: -0.0149692, upper bound: 0.0154751

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.57 + 34.72 = 38.29 seconds
