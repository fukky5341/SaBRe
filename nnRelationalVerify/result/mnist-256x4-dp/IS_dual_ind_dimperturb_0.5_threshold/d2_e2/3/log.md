## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0125892


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0005529, 0.0008407, -0.0005529, 0.0008407, -0.0012660, 0.0012660)
1: (-0.0012680, 0.0026100, -0.0012680, 0.0026100, -0.0038780, 0.0038780)
2: (0.0124312, 0.0182390, 0.0124312, 0.0182390, -0.0058078, 0.0058078)
3: (-0.0012791, 0.0030880, -0.0012791, 0.0030880, -0.0043557, 0.0043557)
4: (-0.0055595, -0.0015312, -0.0055595, -0.0015312, -0.0040283, 0.0040283)
5: (0.0066614, 0.0110206, 0.0066614, 0.0110206, -0.0043400, 0.0043400)
6: (0.0079200, 0.0104191, 0.0079200, 0.0104191, -0.0024991, 0.0024991)
7: (-0.0223239, -0.0128606, -0.0223239, -0.0128606, -0.0086270, 0.0086270)
8: (0.9598302, 0.9869439, 0.9598302, 0.9869439, -0.0271137, 0.0271137)
9: (0.0014903, 0.0094591, 0.0014903, 0.0094591, -0.0074913, 0.0074913)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 1.91 = 3.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0174378, upper bound: 0.0174378

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163896, upper bound: 0.0169820
time: 0.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169820, upper bound: 0.0169820
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 8, lower bound: -0.0163896, upper bound: 0.0169820
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 8, lower bound: -0.0169820, upper bound: 0.0169820

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0004995, 0.0008309, -0.0005510, 0.0008404, -0.0011877, 0.0012532
1: -0.0010181, 0.0025951, -0.0012594, 0.0026095, -0.0035512, 0.0038544
2: 0.0124536, 0.0178646, 0.0124319, 0.0182261, -0.0057724, 0.0051439
3: -0.0012623, 0.0028065, -0.0012786, 0.0030783, -0.0043254, 0.0037953
4: -0.0055440, -0.0017909, -0.0055590, -0.0015402, -0.0040038, 0.0037682
5: 0.0066782, 0.0107396, 0.0066619, 0.0110109, -0.0043098, 0.0037816
6: 0.0079427, 0.0104127, 0.0079207, 0.0104189, -0.0024762, 0.0024920
7: -0.0217139, -0.0128970, -0.0223028, -0.0128617, -0.0075549, 0.0085609
8: 0.9615780, 0.9868394, 0.9598907, 0.9869405, -0.0241875, 0.0269487
9: 0.0015211, 0.0089454, 0.0014913, 0.0094413, -0.0074363, 0.0065575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163896, upper bound: 0.0163896
time: 0.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163896, upper bound: 0.0169820
time: 0.84 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0005250, 0.0009730, -0.0005501, 0.0008402, -0.0012201, 0.0014151
1: -0.0011375, 0.0028127, -0.0012550, 0.0026093, -0.0037467, 0.0040678
2: 0.0121276, 0.0180435, 0.0124323, 0.0182195, -0.0060919, 0.0056111
3: -0.0015075, 0.0029410, -0.0012783, 0.0030734, -0.0045809, 0.0042084
4: -0.0057701, -0.0016668, -0.0055588, -0.0015447, -0.0042254, 0.0038919
5: 0.0064335, 0.0108738, 0.0066622, 0.0110060, -0.0045725, 0.0041908
6: 0.0076126, 0.0105051, 0.0079211, 0.0104188, -0.0028062, 0.0025839
7: -0.0220053, -0.0123658, -0.0222922, -0.0128624, -0.0081351, 0.0092876
8: 0.9607431, 0.9883613, 0.9599210, 0.9869387, -0.0261956, 0.0284403
9: 0.0010737, 0.0091908, 0.0014918, 0.0094324, -0.0080471, 0.0071138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169820, upper bound: 0.0163896
time: 0.89 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0169820, upper bound: 0.0169820
time: 0.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.06 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 8, lower bound: -0.0163896, upper bound: 0.0163896
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 8, lower bound: -0.0163896, upper bound: 0.0169820
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 8, lower bound: -0.0169820, upper bound: 0.0163896
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 8, lower bound: -0.0169820, upper bound: 0.0169820

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004995, 0.0008309, -0.0004995, 0.0008309, -0.0011781, 0.0011781
1: -0.0010181, 0.0025951, -0.0010181, 0.0025951, -0.0035368, 0.0035368
2: 0.0124536, 0.0178646, 0.0124536, 0.0178646, -0.0051224, 0.0051224
3: -0.0012623, 0.0028065, -0.0012623, 0.0028065, -0.0037791, 0.0037791
4: -0.0055440, -0.0017909, -0.0055440, -0.0017909, -0.0037531, 0.0037531
5: 0.0066782, 0.0107396, 0.0066782, 0.0107396, -0.0037655, 0.0037655
6: 0.0079427, 0.0104127, 0.0079427, 0.0104127, -0.0024700, 0.0024700
7: -0.0217139, -0.0128970, -0.0217139, -0.0128970, -0.0075198, 0.0075198
8: 0.9615780, 0.9868394, 0.9615780, 0.9868394, -0.0240869, 0.0240869
9: 0.0015211, 0.0089454, 0.0015211, 0.0089454, -0.0065280, 0.0065280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160373, upper bound: 0.0160725
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160489, upper bound: 0.0161028
time: 0.91 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004995, 0.0008309, -0.0005250, 0.0009730, -0.0013363, 0.0012394
1: -0.0010181, 0.0025951, -0.0011375, 0.0028127, -0.0038308, 0.0037325
2: 0.0124536, 0.0178646, 0.0121276, 0.0180435, -0.0055898, 0.0055726
3: -0.0012623, 0.0028065, -0.0015075, 0.0029410, -0.0042033, 0.0041176
4: -0.0055440, -0.0017909, -0.0057701, -0.0016668, -0.0038771, 0.0039792
5: 0.0066782, 0.0107396, 0.0064335, 0.0108738, -0.0041957, 0.0041034
6: 0.0079427, 0.0104127, 0.0076126, 0.0105051, -0.0025624, 0.0028002
7: -0.0217139, -0.0128970, -0.0220053, -0.0123658, -0.0082535, 0.0084205
8: 0.9615780, 0.9868394, 0.9607431, 0.9883613, -0.0261889, 0.0260962
9: 0.0015211, 0.0089454, 0.0010737, 0.0091908, -0.0073222, 0.0071457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160373, upper bound: 0.0165995
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160489, upper bound: 0.0166532
time: 0.87 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005250, 0.0009730, -0.0004995, 0.0008309, -0.0012394, 0.0013363
1: -0.0011375, 0.0028127, -0.0010181, 0.0025951, -0.0037325, 0.0038308
2: 0.0121276, 0.0180435, 0.0124536, 0.0178646, -0.0055726, 0.0055898
3: -0.0015075, 0.0029410, -0.0012623, 0.0028065, -0.0041176, 0.0042033
4: -0.0057701, -0.0016668, -0.0055440, -0.0017909, -0.0039792, 0.0038771
5: 0.0064335, 0.0108738, 0.0066782, 0.0107396, -0.0041034, 0.0041957
6: 0.0076126, 0.0105051, 0.0079427, 0.0104127, -0.0028002, 0.0025624
7: -0.0220053, -0.0123658, -0.0217139, -0.0128970, -0.0084205, 0.0082535
8: 0.9607431, 0.9883613, 0.9615780, 0.9868394, -0.0260962, 0.0261889
9: 0.0010737, 0.0091908, 0.0015211, 0.0089454, -0.0071457, 0.0073222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166351, upper bound: 0.0160348
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166532, upper bound: 0.0160489
time: 1.03 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005250, 0.0009730, -0.0005250, 0.0009730, -0.0013295, 0.0013295
1: -0.0011375, 0.0028127, -0.0011375, 0.0028127, -0.0039502, 0.0039502
2: 0.0121276, 0.0180435, 0.0121276, 0.0180435, -0.0058471, 0.0058471
3: -0.0015075, 0.0029410, -0.0015075, 0.0029410, -0.0042912, 0.0042912
4: -0.0057701, -0.0016668, -0.0057701, -0.0016668, -0.0041033, 0.0041033
5: 0.0064335, 0.0108738, 0.0064335, 0.0108738, -0.0042735, 0.0042735
6: 0.0076126, 0.0105051, 0.0076126, 0.0105051, -0.0028925, 0.0028925
7: -0.0220053, -0.0123658, -0.0220053, -0.0123658, -0.0083146, 0.0083146
8: 0.9607431, 0.9883613, 0.9607431, 0.9883613, -0.0275455, 0.0275454
9: 0.0010737, 0.0091908, 0.0010737, 0.0091908, -0.0072649, 0.0072649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166351, upper bound: 0.0160461
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0166532, upper bound: 0.0160571
time: 1.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.31 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0160373, upper bound: 0.0160725
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0160489, upper bound: 0.0161028
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0160373, upper bound: 0.0165995
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0160489, upper bound: 0.0166532
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0166351, upper bound: 0.0160348
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0166532, upper bound: 0.0160489
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0166351, upper bound: 0.0160461
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 8, lower bound: -0.0166532, upper bound: 0.0160571

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005061, 0.0007360, -0.0004987, 0.0008163, -0.0011659, 0.0010819
1: -0.0010491, 0.0024495, -0.0010145, 0.0025727, -0.0034389, 0.0033562
2: 0.0126716, 0.0179112, 0.0124871, 0.0178593, -0.0048569, 0.0050018
3: -0.0010984, 0.0028416, -0.0012371, 0.0028026, -0.0035814, 0.0037010
4: -0.0053928, -0.0017586, -0.0055208, -0.0017945, -0.0035983, 0.0037622
5: 0.0068418, 0.0107745, 0.0067033, 0.0107356, -0.0035683, 0.0036888
6: 0.0081634, 0.0103510, 0.0079766, 0.0104033, -0.0022398, 0.0023744
7: -0.0217898, -0.0132522, -0.0217053, -0.0129516, -0.0074553, 0.0071072
8: 0.9613605, 0.9858218, 0.9616026, 0.9866830, -0.0234960, 0.0228436
9: 0.0018201, 0.0090093, 0.0015670, 0.0089381, -0.0061763, 0.0064443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148406, upper bound: 0.0153016
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147922, upper bound: 0.0147988
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004974, 0.0007755, -0.0004995, 0.0008309, -0.0011710, 0.0011312
1: -0.0010082, 0.0025101, -0.0010181, 0.0025951, -0.0034415, 0.0035251
2: 0.0125809, 0.0178498, 0.0124536, 0.0178646, -0.0050928, 0.0049960
3: -0.0011666, 0.0027954, -0.0012623, 0.0028065, -0.0037515, 0.0036930
4: -0.0054557, -0.0018011, -0.0055440, -0.0017909, -0.0036648, 0.0037429
5: 0.0067737, 0.0107285, 0.0066782, 0.0107396, -0.0037378, 0.0036804
6: 0.0080716, 0.0103767, 0.0079427, 0.0104127, -0.0023411, 0.0024340
7: -0.0216898, -0.0131044, -0.0217139, -0.0128970, -0.0074184, 0.0074153
8: 0.9616470, 0.9862452, 0.9615780, 0.9868394, -0.0234798, 0.0239626
9: 0.0016957, 0.0089251, 0.0015211, 0.0089454, -0.0064559, 0.0064171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160678, upper bound: 0.0160901
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160678, upper bound: 0.0161029
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005061, 0.0007360, -0.0005241, 0.0009581, -0.0013240, 0.0011239
1: -0.0010491, 0.0024495, -0.0011334, 0.0027899, -0.0037382, 0.0035736
2: 0.0126716, 0.0179112, 0.0121617, 0.0180374, -0.0051855, 0.0054499
3: -0.0010984, 0.0028416, -0.0014818, 0.0029365, -0.0038301, 0.0040380
4: -0.0053928, -0.0017586, -0.0057464, -0.0016710, -0.0037218, 0.0039879
5: 0.0068418, 0.0107745, 0.0064591, 0.0108693, -0.0038167, 0.0040252
6: 0.0081634, 0.0103510, 0.0076471, 0.0104954, -0.0023320, 0.0027039
7: -0.0217898, -0.0132522, -0.0219955, -0.0124215, -0.0081855, 0.0076088
8: 0.9613605, 0.9858218, 0.9607713, 0.9882020, -0.0255881, 0.0243730
9: 0.0018201, 0.0090093, 0.0011206, 0.0091825, -0.0066189, 0.0070592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148155, upper bound: 0.0157681
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147766, upper bound: 0.0152535
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004974, 0.0007755, -0.0005250, 0.0009730, -0.0013292, 0.0011872
1: -0.0010082, 0.0025101, -0.0011375, 0.0028127, -0.0037422, 0.0036475
2: 0.0125809, 0.0178498, 0.0121276, 0.0180435, -0.0054626, 0.0054462
3: -0.0011666, 0.0027954, -0.0015075, 0.0029410, -0.0041076, 0.0040315
4: -0.0054557, -0.0018011, -0.0057701, -0.0016668, -0.0037889, 0.0039690
5: 0.0067737, 0.0107285, 0.0064335, 0.0108738, -0.0041001, 0.0040184
6: 0.0080716, 0.0103767, 0.0076126, 0.0105051, -0.0024335, 0.0027641
7: -0.0216898, -0.0131044, -0.0220053, -0.0123658, -0.0081521, 0.0082738
8: 0.9616470, 0.9862452, 0.9607431, 0.9883613, -0.0255817, 0.0255020
9: 0.0016957, 0.0089251, 0.0010737, 0.0091908, -0.0072149, 0.0070349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160212, upper bound: 0.0166351
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160212, upper bound: 0.0166532
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005337, 0.0008745, -0.0004987, 0.0008163, -0.0012062, 0.0012375
1: -0.0011783, 0.0026619, -0.0010145, 0.0025727, -0.0036599, 0.0036485
2: 0.0123536, 0.0181047, 0.0124871, 0.0178593, -0.0052948, 0.0053338
3: -0.0013376, 0.0029871, -0.0012371, 0.0028026, -0.0039106, 0.0039495
4: -0.0056134, -0.0016244, -0.0055208, -0.0017945, -0.0038188, 0.0038964
5: 0.0066031, 0.0109198, 0.0067033, 0.0107356, -0.0038970, 0.0039366
6: 0.0078414, 0.0104411, 0.0079766, 0.0104033, -0.0025619, 0.0024645
7: -0.0221051, -0.0127340, -0.0217053, -0.0129516, -0.0079369, 0.0078206
8: 0.9604574, 0.9873065, 0.9616026, 0.9866830, -0.0250470, 0.0248876
9: 0.0013838, 0.0092748, 0.0015670, 0.0089381, -0.0067770, 0.0068754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153038, upper bound: 0.0152849
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0147818
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005227, 0.0009177, -0.0004995, 0.0008309, -0.0012131, 0.0012891
1: -0.0011265, 0.0027280, -0.0010181, 0.0025951, -0.0036626, 0.0037461
2: 0.0122545, 0.0180271, 0.0124536, 0.0178646, -0.0055262, 0.0053316
3: -0.0014121, 0.0029287, -0.0012623, 0.0028065, -0.0040775, 0.0039456
4: -0.0056821, -0.0016782, -0.0055440, -0.0017909, -0.0038912, 0.0038658
5: 0.0065287, 0.0108615, 0.0066782, 0.0107396, -0.0040631, 0.0039325
6: 0.0077410, 0.0104692, 0.0079427, 0.0104127, -0.0026717, 0.0025264
7: -0.0219786, -0.0125726, -0.0217139, -0.0128970, -0.0079205, 0.0081215
8: 0.9608197, 0.9877691, 0.9615780, 0.9868394, -0.0250431, 0.0259860
9: 0.0012478, 0.0091683, 0.0015211, 0.0089454, -0.0070506, 0.0068632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160373
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160489
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005337, 0.0008745, -0.0005241, 0.0009581, -0.0013052, 0.0012174
1: -0.0011783, 0.0026619, -0.0011334, 0.0027899, -0.0037259, 0.0036439
2: 0.0123536, 0.0181047, 0.0121617, 0.0180374, -0.0052401, 0.0053906
3: -0.0013376, 0.0029871, -0.0014818, 0.0029365, -0.0038501, 0.0039768
4: -0.0056134, -0.0016244, -0.0057464, -0.0016710, -0.0039424, 0.0041221
5: 0.0066031, 0.0109198, 0.0064591, 0.0108693, -0.0038350, 0.0039630
6: 0.0078414, 0.0104411, 0.0076471, 0.0104954, -0.0026540, 0.0027939
7: -0.0221051, -0.0127340, -0.0219955, -0.0124215, -0.0079442, 0.0075787
8: 0.9604574, 0.9873065, 0.9607713, 0.9882020, -0.0253522, 0.0246770
9: 0.0013838, 0.0092748, 0.0011206, 0.0091825, -0.0065966, 0.0068819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153038, upper bound: 0.0153458
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0149416
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005227, 0.0009177, -0.0005250, 0.0009730, -0.0013099, 0.0012811
1: -0.0011265, 0.0027280, -0.0011375, 0.0028127, -0.0037286, 0.0038655
2: 0.0122545, 0.0180271, 0.0121276, 0.0180435, -0.0057890, 0.0053831
3: -0.0014121, 0.0029287, -0.0015075, 0.0029410, -0.0042583, 0.0039647
4: -0.0056821, -0.0016782, -0.0057701, -0.0016668, -0.0040153, 0.0040919
5: 0.0065287, 0.0108615, 0.0064335, 0.0108738, -0.0042406, 0.0039500
6: 0.0077410, 0.0104692, 0.0076126, 0.0105051, -0.0027640, 0.0028566
7: -0.0219786, -0.0125726, -0.0220053, -0.0123658, -0.0078993, 0.0082077
8: 0.9608197, 0.9877691, 0.9607431, 0.9883613, -0.0253300, 0.0270259
9: 0.0012478, 0.0091683, 0.0010737, 0.0091908, -0.0071882, 0.0068455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160431
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160571
time: 1.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.25 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0148406, upper bound: 0.0153016
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0147922, upper bound: 0.0147988
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0160678, upper bound: 0.0160901
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0160678, upper bound: 0.0161029
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0148155, upper bound: 0.0157681
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0147766, upper bound: 0.0152535
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0160212, upper bound: 0.0166351
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0160212, upper bound: 0.0166532
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0153038, upper bound: 0.0152849
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0147818
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160373
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160489
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0153038, upper bound: 0.0153458
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0149416
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160431
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 8, lower bound: -0.0165984, upper bound: 0.0160571

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005061, 0.0007360, -0.0004822, 0.0008115, -0.0011609, 0.0010628
1: -0.0010491, 0.0024495, -0.0009372, 0.0025653, -0.0034314, 0.0032708
2: 0.0126716, 0.0179112, 0.0124981, 0.0177436, -0.0047249, 0.0049906
3: -0.0010984, 0.0028416, -0.0012288, 0.0027155, -0.0034802, 0.0036925
4: -0.0053928, -0.0017586, -0.0055131, -0.0018749, -0.0035180, 0.0037545
5: 0.0068418, 0.0107745, 0.0067116, 0.0106487, -0.0034673, 0.0036804
6: 0.0081634, 0.0103510, 0.0079878, 0.0104001, -0.0022367, 0.0023632
7: -0.0217898, -0.0132522, -0.0215166, -0.0129696, -0.0074370, 0.0068842
8: 0.9613605, 0.9858218, 0.9621431, 0.9866316, -0.0234435, 0.0222312
9: 0.0018201, 0.0090093, 0.0015821, 0.0087793, -0.0059871, 0.0064289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147922, upper bound: 0.0147988
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147922, upper bound: 0.0147988
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004970, 0.0007333, -0.0004491, 0.0009253, -0.0012638, 0.0010506
1: -0.0010064, 0.0024455, -0.0007821, 0.0027397, -0.0036924, 0.0032276
2: 0.0126776, 0.0178472, 0.0122370, 0.0175112, -0.0047674, 0.0053505
3: -0.0010938, 0.0027935, -0.0014252, 0.0025408, -0.0034948, 0.0039447
4: -0.0053886, -0.0018029, -0.0056942, -0.0020360, -0.0033526, 0.0038913
5: 0.0068463, 0.0107265, 0.0065156, 0.0104744, -0.0034802, 0.0039303
6: 0.0081696, 0.0103493, 0.0077234, 0.0104741, -0.0023045, 0.0026259
7: -0.0216856, -0.0132621, -0.0211381, -0.0125441, -0.0078484, 0.0067842
8: 0.9616591, 0.9857935, 0.9632276, 0.9878505, -0.0251620, 0.0224706
9: 0.0018284, 0.0089216, 0.0012239, 0.0084606, -0.0059323, 0.0067958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147060, upper bound: 0.0146205
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147102, upper bound: 0.0147098
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004974, 0.0007755, -0.0005061, 0.0007360, -0.0010763, 0.0011268
1: -0.0010082, 0.0025101, -0.0010491, 0.0024495, -0.0032809, 0.0033721
2: 0.0125809, 0.0178498, 0.0126716, 0.0179112, -0.0049017, 0.0047554
3: -0.0011666, 0.0027954, -0.0010984, 0.0028416, -0.0036257, 0.0035121
4: -0.0054557, -0.0018011, -0.0053928, -0.0017586, -0.0036971, 0.0035917
5: 0.0067737, 0.0107285, 0.0068418, 0.0107745, -0.0036137, 0.0034999
6: 0.0080716, 0.0103767, 0.0081634, 0.0103510, -0.0022794, 0.0022133
7: -0.0216898, -0.0131044, -0.0217898, -0.0132522, -0.0070264, 0.0072922
8: 0.9616470, 0.9862452, 0.9613605, 0.9858218, -0.0223567, 0.0230289
9: 0.0016957, 0.0089251, 0.0018201, 0.0090093, -0.0063071, 0.0060870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153016, upper bound: 0.0148406
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0147922
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004974, 0.0007755, -0.0004974, 0.0007755, -0.0011261, 0.0011261
1: -0.0010082, 0.0025101, -0.0010082, 0.0025101, -0.0034598, 0.0034598
2: 0.0125809, 0.0178498, 0.0125809, 0.0178498, -0.0050103, 0.0050103
3: -0.0011666, 0.0027954, -0.0011666, 0.0027954, -0.0036976, 0.0036976
4: -0.0054557, -0.0018011, -0.0054557, -0.0018011, -0.0036546, 0.0036546
5: 0.0067737, 0.0107285, 0.0067737, 0.0107285, -0.0036844, 0.0036844
6: 0.0080716, 0.0103767, 0.0080716, 0.0103767, -0.0023051, 0.0023051
7: -0.0216898, -0.0131044, -0.0216898, -0.0131044, -0.0073442, 0.0073442
8: 0.9616470, 0.9862452, 0.9616470, 0.9862452, -0.0235633, 0.0235634
9: 0.0016957, 0.0089251, 0.0016957, 0.0089251, -0.0063834, 0.0063834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153016, upper bound: 0.0148406
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0149247
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005061, 0.0007360, -0.0005054, 0.0009532, -0.0013188, 0.0011041
1: -0.0010491, 0.0024495, -0.0010458, 0.0027825, -0.0037305, 0.0034821
2: 0.0126716, 0.0179112, 0.0121729, 0.0179062, -0.0050466, 0.0054384
3: -0.0010984, 0.0028416, -0.0014734, 0.0028378, -0.0037241, 0.0040293
4: -0.0053928, -0.0017586, -0.0057387, -0.0017621, -0.0036308, 0.0039801
5: 0.0068418, 0.0107745, 0.0064674, 0.0107708, -0.0037109, 0.0040166
6: 0.0081634, 0.0103510, 0.0076584, 0.0104923, -0.0023288, 0.0026926
7: -0.0217898, -0.0132522, -0.0217816, -0.0124396, -0.0081667, 0.0073781
8: 0.9613605, 0.9858218, 0.9613839, 0.9881501, -0.0255345, 0.0237263
9: 0.0018201, 0.0090093, 0.0011358, 0.0090024, -0.0064239, 0.0070435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147766, upper bound: 0.0152535
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147766, upper bound: 0.0152535
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004970, 0.0007333, -0.0004746, 0.0010688, -0.0014326, 0.0010916
1: -0.0010064, 0.0024455, -0.0009015, 0.0029596, -0.0039121, 0.0033469
2: 0.0126776, 0.0178472, 0.0119076, 0.0176900, -0.0050124, 0.0057077
3: -0.0010938, 0.0027935, -0.0016729, 0.0026753, -0.0037285, 0.0042318
4: -0.0053886, -0.0018029, -0.0059227, -0.0019120, -0.0034766, 0.0041197
5: 0.0068463, 0.0107265, 0.0062684, 0.0106085, -0.0037139, 0.0042187
6: 0.0081696, 0.0103493, 0.0073898, 0.0105674, -0.0023978, 0.0029595
7: -0.0216856, -0.0132621, -0.0214294, -0.0120074, -0.0086028, 0.0072753
8: 0.9616591, 0.9857935, 0.9623931, 0.9893883, -0.0267927, 0.0234004
9: 0.0018284, 0.0089216, 0.0007719, 0.0087059, -0.0063642, 0.0074118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146890, upper bound: 0.0150401
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146927, upper bound: 0.0151372
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004974, 0.0007755, -0.0005337, 0.0008745, -0.0012320, 0.0011672
1: -0.0010082, 0.0025101, -0.0011783, 0.0026619, -0.0035732, 0.0035930
2: 0.0125809, 0.0178498, 0.0123536, 0.0181047, -0.0052337, 0.0051933
3: -0.0011666, 0.0027954, -0.0013376, 0.0029871, -0.0038743, 0.0038413
4: -0.0054557, -0.0018011, -0.0056134, -0.0016244, -0.0038313, 0.0038123
5: 0.0067737, 0.0107285, 0.0066031, 0.0109198, -0.0038615, 0.0038285
6: 0.0080716, 0.0103767, 0.0078414, 0.0104411, -0.0023695, 0.0025353
7: -0.0216898, -0.0131044, -0.0221051, -0.0127340, -0.0077398, 0.0077739
8: 0.9616470, 0.9862452, 0.9604574, 0.9873065, -0.0244006, 0.0245799
9: 0.0016957, 0.0089251, 0.0013838, 0.0092748, -0.0067382, 0.0066878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152849, upper bound: 0.0153038
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0152391
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004974, 0.0007755, -0.0005227, 0.0009177, -0.0012840, 0.0011650
1: -0.0010082, 0.0025101, -0.0011265, 0.0027280, -0.0037362, 0.0036366
2: 0.0125809, 0.0178498, 0.0122545, 0.0180271, -0.0053140, 0.0054437
3: -0.0011666, 0.0027954, -0.0014121, 0.0029287, -0.0039241, 0.0040235
4: -0.0054557, -0.0018011, -0.0056821, -0.0016782, -0.0037775, 0.0038810
5: 0.0067737, 0.0107285, 0.0065287, 0.0108615, -0.0039104, 0.0040097
6: 0.0080716, 0.0103767, 0.0077410, 0.0104692, -0.0023975, 0.0026356
7: -0.0216898, -0.0131044, -0.0219786, -0.0125726, -0.0080505, 0.0078044
8: 0.9616470, 0.9862452, 0.9608197, 0.9877691, -0.0255868, 0.0249802
9: 0.0016957, 0.0089251, 0.0012478, 0.0091683, -0.0067813, 0.0069781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152849, upper bound: 0.0154603
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0153901
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005337, 0.0008745, -0.0004822, 0.0008115, -0.0012012, 0.0012184
1: -0.0011783, 0.0026619, -0.0009372, 0.0025653, -0.0036524, 0.0035632
2: 0.0123536, 0.0181047, 0.0124981, 0.0177436, -0.0051627, 0.0053225
3: -0.0013376, 0.0029871, -0.0012288, 0.0027155, -0.0038094, 0.0039411
4: -0.0056134, -0.0016244, -0.0055131, -0.0018749, -0.0037385, 0.0038888
5: 0.0066031, 0.0109198, 0.0067116, 0.0106487, -0.0037959, 0.0039282
6: 0.0078414, 0.0104411, 0.0079878, 0.0104001, -0.0025588, 0.0024533
7: -0.0221051, -0.0127340, -0.0215166, -0.0129696, -0.0079186, 0.0075976
8: 0.9604574, 0.9873065, 0.9621431, 0.9866316, -0.0249945, 0.0242752
9: 0.0013838, 0.0092748, 0.0015821, 0.0087793, -0.0065878, 0.0068600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0147818
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0147818
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005239, 0.0008719, -0.0004491, 0.0009253, -0.0013057, 0.0012063
1: -0.0011325, 0.0026579, -0.0007821, 0.0027397, -0.0038722, 0.0034400
2: 0.0123595, 0.0180360, 0.0122370, 0.0175112, -0.0051518, 0.0056465
3: -0.0013331, 0.0029354, -0.0014252, 0.0025408, -0.0038241, 0.0041714
4: -0.0056093, -0.0016720, -0.0056942, -0.0020360, -0.0035733, 0.0040222
5: 0.0066075, 0.0108682, 0.0065156, 0.0104744, -0.0038088, 0.0041569
6: 0.0078474, 0.0104394, 0.0077234, 0.0104741, -0.0026267, 0.0027160
7: -0.0219932, -0.0127437, -0.0211381, -0.0125441, -0.0083388, 0.0074976
8: 0.9607778, 0.9872789, 0.9632276, 0.9878505, -0.0265364, 0.0240512
9: 0.0013919, 0.0091806, 0.0012239, 0.0084606, -0.0065330, 0.0072192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0146014
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0146835
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005227, 0.0009177, -0.0005061, 0.0007360, -0.0011183, 0.0012840
1: -0.0011265, 0.0027280, -0.0010491, 0.0024495, -0.0035020, 0.0036736
2: 0.0122545, 0.0180271, 0.0126716, 0.0179112, -0.0053533, 0.0050910
3: -0.0014121, 0.0029287, -0.0010984, 0.0028416, -0.0039653, 0.0037647
4: -0.0056821, -0.0016782, -0.0053928, -0.0017586, -0.0039235, 0.0037146
5: 0.0065287, 0.0108615, 0.0068418, 0.0107745, -0.0039527, 0.0037519
6: 0.0077410, 0.0104692, 0.0081634, 0.0103510, -0.0026100, 0.0023057
7: -0.0219786, -0.0125726, -0.0217898, -0.0132522, -0.0075285, 0.0080280
8: 0.9608197, 0.9877691, 0.9613605, 0.9858218, -0.0239200, 0.0251371
9: 0.0012478, 0.0091683, 0.0018201, 0.0090093, -0.0069267, 0.0065331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0148155
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0147766
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005227, 0.0009177, -0.0004974, 0.0007755, -0.0011650, 0.0012840
1: -0.0011265, 0.0027280, -0.0010082, 0.0025101, -0.0036366, 0.0037362
2: 0.0122545, 0.0180271, 0.0125809, 0.0178498, -0.0054437, 0.0053140
3: -0.0014121, 0.0029287, -0.0011666, 0.0027954, -0.0040235, 0.0039241
4: -0.0056821, -0.0016782, -0.0054557, -0.0018011, -0.0038810, 0.0037775
5: 0.0065287, 0.0108615, 0.0067737, 0.0107285, -0.0040097, 0.0039104
6: 0.0077410, 0.0104692, 0.0080716, 0.0103767, -0.0026356, 0.0023975
7: -0.0219786, -0.0125726, -0.0216898, -0.0131044, -0.0078044, 0.0080505
8: 0.9608197, 0.9877691, 0.9616470, 0.9862452, -0.0249802, 0.0255868
9: 0.0012478, 0.0091683, 0.0016957, 0.0089251, -0.0069781, 0.0067813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0149371
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149049
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005337, 0.0008745, -0.0005054, 0.0009532, -0.0013002, 0.0011969
1: -0.0011783, 0.0026619, -0.0010458, 0.0027825, -0.0037185, 0.0035536
2: 0.0123536, 0.0181047, 0.0121729, 0.0179062, -0.0050965, 0.0053794
3: -0.0013376, 0.0029871, -0.0014734, 0.0028378, -0.0037402, 0.0039684
4: -0.0056134, -0.0016244, -0.0057387, -0.0017621, -0.0038513, 0.0041144
5: 0.0066031, 0.0109198, 0.0064674, 0.0107708, -0.0037251, 0.0039546
6: 0.0078414, 0.0104411, 0.0076584, 0.0104923, -0.0026509, 0.0027827
7: -0.0221051, -0.0127340, -0.0217816, -0.0124396, -0.0079259, 0.0073387
8: 0.9604574, 0.9873065, 0.9613839, 0.9881501, -0.0252998, 0.0240154
9: 0.0013838, 0.0092748, 0.0011358, 0.0090024, -0.0063931, 0.0068665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0149416
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0149416
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005239, 0.0008719, -0.0004746, 0.0010688, -0.0014146, 0.0011864
1: -0.0011325, 0.0026579, -0.0009015, 0.0029596, -0.0039233, 0.0035594
2: 0.0123595, 0.0180360, 0.0119076, 0.0176900, -0.0051632, 0.0056821
3: -0.0013331, 0.0029354, -0.0016729, 0.0026753, -0.0037711, 0.0041938
4: -0.0056093, -0.0016720, -0.0059227, -0.0019120, -0.0036973, 0.0042507
5: 0.0066075, 0.0108682, 0.0062684, 0.0106085, -0.0037540, 0.0041793
6: 0.0078474, 0.0104394, 0.0073898, 0.0105674, -0.0027200, 0.0030496
7: -0.0219932, -0.0127437, -0.0214294, -0.0120074, -0.0084077, 0.0072573
8: 0.9607778, 0.9872789, 0.9623931, 0.9893883, -0.0267172, 0.0243757
9: 0.0013919, 0.0091806, 0.0007719, 0.0087059, -0.0063545, 0.0072748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0146852
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0147912
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005227, 0.0009177, -0.0005337, 0.0008745, -0.0012117, 0.0012657
1: -0.0011265, 0.0027280, -0.0011783, 0.0026619, -0.0035639, 0.0036591
2: 0.0122545, 0.0180271, 0.0123536, 0.0181047, -0.0052905, 0.0051364
3: -0.0014121, 0.0029287, -0.0013376, 0.0029871, -0.0039016, 0.0037792
4: -0.0056821, -0.0016782, -0.0056134, -0.0016244, -0.0040577, 0.0039352
5: 0.0065287, 0.0108615, 0.0066031, 0.0109198, -0.0038879, 0.0037648
6: 0.0077410, 0.0104692, 0.0078414, 0.0104411, -0.0027000, 0.0026278
7: -0.0219786, -0.0125726, -0.0221051, -0.0127340, -0.0074973, 0.0077811
8: 0.9608197, 0.9877691, 0.9604574, 0.9873065, -0.0241781, 0.0248849
9: 0.0012478, 0.0091683, 0.0013838, 0.0092748, -0.0067446, 0.0065070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0150152
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149715
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005227, 0.0009177, -0.0005227, 0.0009177, -0.0012637, 0.0012637
1: -0.0011265, 0.0027280, -0.0011265, 0.0027280, -0.0037439, 0.0037439
2: 0.0122545, 0.0180271, 0.0122545, 0.0180271, -0.0053939, 0.0053939
3: -0.0014121, 0.0029287, -0.0014121, 0.0029287, -0.0039665, 0.0039665
4: -0.0056821, -0.0016782, -0.0056821, -0.0016782, -0.0040039, 0.0040039
5: 0.0065287, 0.0108615, 0.0065287, 0.0108615, -0.0039513, 0.0039513
6: 0.0077410, 0.0104692, 0.0077410, 0.0104692, -0.0027281, 0.0027281
7: -0.0219786, -0.0125726, -0.0219786, -0.0125726, -0.0078052, 0.0078052
8: 0.9608197, 0.9877691, 0.9608197, 0.9877691, -0.0253933, 0.0253933
9: 0.0012478, 0.0091683, 0.0012478, 0.0091683, -0.0067995, 0.0067995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0151318
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0150970
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.17 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147922, upper bound: 0.0147988
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147922, upper bound: 0.0147988
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147060, upper bound: 0.0146205
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147102, upper bound: 0.0147098
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0153016, upper bound: 0.0148406
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0147922
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0153016, upper bound: 0.0148406
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0149247
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147766, upper bound: 0.0152535
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147766, upper bound: 0.0152535
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0146890, upper bound: 0.0150401
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0146927, upper bound: 0.0151372
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152849, upper bound: 0.0153038
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0152391
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152849, upper bound: 0.0154603
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0153901
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0147818
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0147818
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0146014
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0146835
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0148155
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0147766
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0149371
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149049
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0149416
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152391, upper bound: 0.0149416
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0146852
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0151566, upper bound: 0.0147912
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0150152
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149715
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0157681, upper bound: 0.0151318
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.17
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0150970

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004894, 0.0007313, -0.0004822, 0.0008115, -0.0011407, 0.0010579
1: -0.0009709, 0.0024423, -0.0009372, 0.0025653, -0.0033390, 0.0032636
2: 0.0126823, 0.0177941, 0.0124981, 0.0177436, -0.0047141, 0.0048492
3: -0.0010903, 0.0027535, -0.0012288, 0.0027155, -0.0034721, 0.0035837
4: -0.0053853, -0.0018398, -0.0055131, -0.0018749, -0.0035105, 0.0036733
5: 0.0068498, 0.0106867, 0.0067116, 0.0106487, -0.0034592, 0.0035715
6: 0.0081743, 0.0103480, 0.0079878, 0.0104001, -0.0022258, 0.0023602
7: -0.0215990, -0.0132697, -0.0215166, -0.0129696, -0.0072017, 0.0068666
8: 0.9619073, 0.9857716, 0.9621431, 0.9866316, -0.0227879, 0.0221809
9: 0.0018349, 0.0088486, 0.0015821, 0.0087793, -0.0059723, 0.0062277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0151188
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147583, upper bound: 0.0151188
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004595, 0.0008509, -0.0004822, 0.0008115, -0.0011265, 0.0011742
1: -0.0008308, 0.0026257, -0.0009372, 0.0025653, -0.0033803, 0.0035629
2: 0.0124077, 0.0175842, 0.0124981, 0.0177436, -0.0051592, 0.0048746
3: -0.0012968, 0.0025957, -0.0012288, 0.0027155, -0.0037913, 0.0035859
4: -0.0055758, -0.0019854, -0.0055131, -0.0018749, -0.0037009, 0.0035278
5: 0.0066437, 0.0105291, 0.0067116, 0.0106487, -0.0037762, 0.0035721
6: 0.0078963, 0.0104257, 0.0079878, 0.0104001, -0.0025039, 0.0024380
7: -0.0212570, -0.0128223, -0.0215166, -0.0129696, -0.0070688, 0.0074080
8: 0.9628871, 0.9870535, 0.9621431, 0.9866316, -0.0229478, 0.0242950
9: 0.0014581, 0.0085607, 0.0015821, 0.0087793, -0.0064556, 0.0061524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0151188
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147583, upper bound: 0.0151188
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004967, 0.0006573, -0.0004484, 0.0009145, -0.0012532, 0.0009771
1: -0.0010051, 0.0023289, -0.0007791, 0.0027231, -0.0036150, 0.0031080
2: 0.0128522, 0.0178452, 0.0122619, 0.0175067, -0.0045785, 0.0052582
3: -0.0009626, 0.0027919, -0.0014065, 0.0025374, -0.0033547, 0.0038854
4: -0.0052675, -0.0018043, -0.0056770, -0.0020391, -0.0032284, 0.0038726
5: 0.0069773, 0.0107250, 0.0065343, 0.0104710, -0.0033404, 0.0038719
6: 0.0083463, 0.0102999, 0.0077486, 0.0104670, -0.0021207, 0.0025513
7: -0.0216823, -0.0135465, -0.0211308, -0.0125847, -0.0077922, 0.0064966
8: 0.9616687, 0.9849786, 0.9632487, 0.9877343, -0.0247072, 0.0215851
9: 0.0020679, 0.0089188, 0.0012580, 0.0084544, -0.0056853, 0.0067326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0145914
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0146205
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004949, 0.0006959, -0.0004491, 0.0009253, -0.0012619, 0.0010133
1: -0.0009967, 0.0023881, -0.0007821, 0.0027397, -0.0036408, 0.0031702
2: 0.0127636, 0.0178327, 0.0122370, 0.0175112, -0.0047249, 0.0052877
3: -0.0010292, 0.0027826, -0.0014252, 0.0025408, -0.0034557, 0.0039044
4: -0.0053290, -0.0018130, -0.0056942, -0.0020360, -0.0032930, 0.0038812
5: 0.0069108, 0.0107157, 0.0065156, 0.0104744, -0.0034407, 0.0038908
6: 0.0082566, 0.0103249, 0.0077234, 0.0104741, -0.0022175, 0.0026016
7: -0.0216620, -0.0134021, -0.0211381, -0.0125441, -0.0078201, 0.0066600
8: 0.9617269, 0.9853923, 0.9632276, 0.9878505, -0.0248544, 0.0221647
9: 0.0019463, 0.0089017, 0.0012239, 0.0084606, -0.0058362, 0.0067613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147066
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147098
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0005061, 0.0007360, -0.0010573, 0.0011218
1: -0.0009311, 0.0025027, -0.0010491, 0.0024495, -0.0031928, 0.0033645
2: 0.0125920, 0.0177344, 0.0126716, 0.0179112, -0.0048904, 0.0046223
3: -0.0011583, 0.0027086, -0.0010984, 0.0028416, -0.0036172, 0.0034093
4: -0.0054480, -0.0018812, -0.0053928, -0.0017586, -0.0036895, 0.0035116
5: 0.0067820, 0.0106419, 0.0068418, 0.0107745, -0.0036052, 0.0033971
6: 0.0080828, 0.0103736, 0.0081634, 0.0103510, -0.0022682, 0.0022101
7: -0.0215017, -0.0131225, -0.0217898, -0.0132522, -0.0068038, 0.0072737
8: 0.9621859, 0.9861934, 0.9613605, 0.9858218, -0.0217360, 0.0229759
9: 0.0017109, 0.0087667, 0.0018201, 0.0090093, -0.0062915, 0.0058957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0147922
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0147922
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0004970, 0.0007333, -0.0010455, 0.0012235
1: -0.0007766, 0.0026746, -0.0010064, 0.0024455, -0.0032220, 0.0036204
2: 0.0123345, 0.0175030, 0.0126776, 0.0178472, -0.0052426, 0.0046602
3: -0.0013519, 0.0025346, -0.0010938, 0.0027935, -0.0038635, 0.0034213
4: -0.0056266, -0.0020417, -0.0053886, -0.0018029, -0.0038237, 0.0033469
5: 0.0065887, 0.0104681, 0.0068463, 0.0107265, -0.0038493, 0.0034077
6: 0.0078221, 0.0104465, 0.0081696, 0.0103493, -0.0025272, 0.0022769
7: -0.0211247, -0.0127029, -0.0216856, -0.0132621, -0.0067073, 0.0076726
8: 0.9632663, 0.9873955, 0.9616591, 0.9857935, -0.0219559, 0.0246582
9: 0.0013576, 0.0084492, 0.0018284, 0.0089216, -0.0066477, 0.0058414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147060
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147098, upper bound: 0.0147102
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0004974, 0.0007755, -0.0011051, 0.0011210
1: -0.0009311, 0.0025027, -0.0010082, 0.0025101, -0.0033649, 0.0034524
2: 0.0125920, 0.0177344, 0.0125809, 0.0178498, -0.0049993, 0.0048657
3: -0.0011583, 0.0027086, -0.0011666, 0.0027954, -0.0036893, 0.0035851
4: -0.0054480, -0.0018812, -0.0054557, -0.0018011, -0.0036469, 0.0035745
5: 0.0067820, 0.0106419, 0.0067737, 0.0107285, -0.0036762, 0.0035720
6: 0.0080828, 0.0103736, 0.0080716, 0.0103767, -0.0022939, 0.0023019
7: -0.0215017, -0.0131225, -0.0216898, -0.0131044, -0.0070976, 0.0073264
8: 0.9621859, 0.9861934, 0.9616470, 0.9862452, -0.0228887, 0.0235123
9: 0.0017109, 0.0087667, 0.0016957, 0.0089251, -0.0063684, 0.0061673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149215, upper bound: 0.0149247
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149215, upper bound: 0.0149247
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0004877, 0.0007729, -0.0010942, 0.0012219
1: -0.0007766, 0.0026746, -0.0009629, 0.0025062, -0.0032827, 0.0036375
2: 0.0123345, 0.0175030, 0.0125867, 0.0177821, -0.0053636, 0.0049075
3: -0.0013519, 0.0025346, -0.0011622, 0.0027445, -0.0039468, 0.0036011
4: -0.0056266, -0.0020417, -0.0054517, -0.0018482, -0.0037785, 0.0034099
5: 0.0065887, 0.0104681, 0.0067781, 0.0106776, -0.0039314, 0.0035863
6: 0.0078221, 0.0104465, 0.0080775, 0.0103750, -0.0025530, 0.0023690
7: -0.0211247, -0.0127029, -0.0215793, -0.0131140, -0.0070118, 0.0077549
8: 0.9632663, 0.9873955, 0.9619636, 0.9862179, -0.0229516, 0.0252399
9: 0.0013576, 0.0084492, 0.0017037, 0.0088321, -0.0067415, 0.0061268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148407, upper bound: 0.0148433
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004894, 0.0007313, -0.0005054, 0.0009532, -0.0012986, 0.0010992
1: -0.0009709, 0.0024423, -0.0010458, 0.0027825, -0.0036381, 0.0034749
2: 0.0126823, 0.0177941, 0.0121729, 0.0179062, -0.0050358, 0.0052970
3: -0.0010903, 0.0027535, -0.0014734, 0.0028378, -0.0037160, 0.0039205
4: -0.0053853, -0.0018398, -0.0057387, -0.0017621, -0.0036233, 0.0038989
5: 0.0068498, 0.0106867, 0.0064674, 0.0107708, -0.0037028, 0.0039077
6: 0.0081743, 0.0103480, 0.0076584, 0.0104923, -0.0023179, 0.0026896
7: -0.0215990, -0.0132697, -0.0217816, -0.0124396, -0.0079315, 0.0073605
8: 0.9619073, 0.9857716, 0.9613839, 0.9881501, -0.0248789, 0.0236760
9: 0.0018349, 0.0088486, 0.0011358, 0.0090024, -0.0064091, 0.0068423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0155721
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147315, upper bound: 0.0155721
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004595, 0.0008509, -0.0005054, 0.0009532, -0.0012845, 0.0012192
1: -0.0008308, 0.0026257, -0.0010458, 0.0027825, -0.0036133, 0.0036715
2: 0.0124077, 0.0175842, 0.0121729, 0.0179062, -0.0054491, 0.0053225
3: -0.0012968, 0.0025957, -0.0014734, 0.0028378, -0.0040150, 0.0039227
4: -0.0055758, -0.0019854, -0.0057387, -0.0017621, -0.0038137, 0.0037534
5: 0.0066437, 0.0105291, 0.0064674, 0.0107708, -0.0040003, 0.0039083
6: 0.0078963, 0.0104257, 0.0076584, 0.0104923, -0.0025960, 0.0027673
7: -0.0212570, -0.0128223, -0.0217816, -0.0124396, -0.0077986, 0.0079311
8: 0.9628871, 0.9870535, 0.9613839, 0.9881501, -0.0250387, 0.0256387
9: 0.0014581, 0.0085607, 0.0011358, 0.0090024, -0.0068991, 0.0067669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0155721
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147315, upper bound: 0.0155721
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004967, 0.0006573, -0.0004740, 0.0010579, -0.0014209, 0.0010182
1: -0.0010051, 0.0023289, -0.0008986, 0.0029429, -0.0038315, 0.0032275
2: 0.0128522, 0.0178452, 0.0119327, 0.0176857, -0.0048335, 0.0056116
3: -0.0009626, 0.0027919, -0.0016540, 0.0026720, -0.0035886, 0.0041674
4: -0.0052675, -0.0018043, -0.0059053, -0.0019150, -0.0033526, 0.0041010
5: 0.0069773, 0.0107250, 0.0062871, 0.0106053, -0.0035745, 0.0041551
6: 0.0083463, 0.0102999, 0.0074151, 0.0105603, -0.0022140, 0.0028847
7: -0.0216823, -0.0135465, -0.0214224, -0.0120482, -0.0085366, 0.0069876
8: 0.9616687, 0.9849786, 0.9624131, 0.9892714, -0.0263235, 0.0225655
9: 0.0020679, 0.0089188, 0.0008063, 0.0086999, -0.0061173, 0.0073319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0149950
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0150401
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004949, 0.0006959, -0.0004746, 0.0010688, -0.0014307, 0.0010554
1: -0.0009967, 0.0023881, -0.0009015, 0.0029596, -0.0038583, 0.0032895
2: 0.0127636, 0.0178327, 0.0119076, 0.0176900, -0.0049264, 0.0056438
3: -0.0010292, 0.0027826, -0.0016729, 0.0026753, -0.0036880, 0.0041910
4: -0.0053290, -0.0018130, -0.0059227, -0.0019120, -0.0034170, 0.0041097
5: 0.0069108, 0.0107157, 0.0062684, 0.0106085, -0.0036734, 0.0041786
6: 0.0082566, 0.0103249, 0.0073898, 0.0105674, -0.0023108, 0.0029351
7: -0.0216620, -0.0134021, -0.0214294, -0.0120074, -0.0085743, 0.0071680
8: 0.9617269, 0.9853923, 0.9623931, 0.9893883, -0.0264788, 0.0229992
9: 0.0019463, 0.0089017, 0.0007719, 0.0087059, -0.0062805, 0.0073720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0005337, 0.0008745, -0.0012129, 0.0011621
1: -0.0009311, 0.0025027, -0.0011783, 0.0026619, -0.0034852, 0.0035855
2: 0.0125920, 0.0177344, 0.0123536, 0.0181047, -0.0052224, 0.0050601
3: -0.0011583, 0.0027086, -0.0013376, 0.0029871, -0.0038658, 0.0037385
4: -0.0054480, -0.0018812, -0.0056134, -0.0016244, -0.0038237, 0.0037322
5: 0.0067820, 0.0106419, 0.0066031, 0.0109198, -0.0038530, 0.0037258
6: 0.0080828, 0.0103736, 0.0078414, 0.0104411, -0.0023583, 0.0025322
7: -0.0215017, -0.0131225, -0.0221051, -0.0127340, -0.0075172, 0.0077554
8: 0.9621859, 0.9861934, 0.9604574, 0.9873065, -0.0237799, 0.0245269
9: 0.0017109, 0.0087667, 0.0013838, 0.0092748, -0.0067226, 0.0064965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0152391
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0152391
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0005239, 0.0008719, -0.0012012, 0.0012655
1: -0.0007766, 0.0026746, -0.0011325, 0.0026579, -0.0034345, 0.0038071
2: 0.0123345, 0.0175030, 0.0123595, 0.0180360, -0.0055386, 0.0050980
3: -0.0013519, 0.0025346, -0.0013331, 0.0029354, -0.0040902, 0.0037505
4: -0.0056266, -0.0020417, -0.0056093, -0.0016720, -0.0039546, 0.0035676
5: 0.0065887, 0.0104681, 0.0066075, 0.0108682, -0.0040759, 0.0037363
6: 0.0078221, 0.0104465, 0.0078474, 0.0104394, -0.0026174, 0.0025991
7: -0.0211247, -0.0127029, -0.0219932, -0.0127437, -0.0074207, 0.0081630
8: 0.9632663, 0.9873955, 0.9607778, 0.9872789, -0.0239999, 0.0260326
9: 0.0013576, 0.0084492, 0.0013919, 0.0091806, -0.0070711, 0.0064421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146835, upper bound: 0.0151566
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0005227, 0.0009177, -0.0012631, 0.0011600
1: -0.0009311, 0.0025027, -0.0011265, 0.0027280, -0.0036543, 0.0036292
2: 0.0125920, 0.0177344, 0.0122545, 0.0180271, -0.0053031, 0.0052991
3: -0.0011583, 0.0027086, -0.0014121, 0.0029287, -0.0039159, 0.0039110
4: -0.0054480, -0.0018812, -0.0056821, -0.0016782, -0.0037698, 0.0038009
5: 0.0067820, 0.0106419, 0.0065287, 0.0108615, -0.0039022, 0.0038973
6: 0.0080828, 0.0103736, 0.0077410, 0.0104692, -0.0023863, 0.0026325
7: -0.0215017, -0.0131225, -0.0219786, -0.0125726, -0.0078039, 0.0077866
8: 0.9621859, 0.9861934, 0.9608197, 0.9877691, -0.0249122, 0.0249291
9: 0.0017109, 0.0087667, 0.0012478, 0.0091683, -0.0067663, 0.0067620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149025, upper bound: 0.0153901
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149025, upper bound: 0.0153901
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0005121, 0.0009151, -0.0012521, 0.0012615
1: -0.0007766, 0.0026746, -0.0010773, 0.0027241, -0.0035007, 0.0037519
2: 0.0123345, 0.0175030, 0.0122603, 0.0179533, -0.0056188, 0.0052426
3: -0.0013519, 0.0025346, -0.0014076, 0.0028732, -0.0041549, 0.0039270
4: -0.0056266, -0.0020417, -0.0056780, -0.0017294, -0.0038973, 0.0036363
5: 0.0065887, 0.0104681, 0.0065331, 0.0108062, -0.0041394, 0.0039116
6: 0.0078221, 0.0104465, 0.0077470, 0.0104675, -0.0026454, 0.0026995
7: -0.0211247, -0.0127029, -0.0218584, -0.0125821, -0.0077178, 0.0082104
8: 0.9632663, 0.9873955, 0.9611638, 0.9877416, -0.0244754, 0.0262316
9: 0.0013576, 0.0084492, 0.0012559, 0.0090671, -0.0071314, 0.0067213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147707, upper bound: 0.0153021
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148207, upper bound: 0.0153021
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005137, 0.0008697, -0.0004822, 0.0008115, -0.0011799, 0.0012133
1: -0.0010848, 0.0026544, -0.0009372, 0.0025653, -0.0035522, 0.0035555
2: 0.0123647, 0.0179645, 0.0124981, 0.0177436, -0.0051512, 0.0051722
3: -0.0013292, 0.0028817, -0.0012288, 0.0027155, -0.0038007, 0.0038271
4: -0.0056057, -0.0017216, -0.0055131, -0.0018749, -0.0037308, 0.0037915
5: 0.0066114, 0.0108146, 0.0067116, 0.0106487, -0.0037873, 0.0038144
6: 0.0078526, 0.0104379, 0.0079878, 0.0104001, -0.0025475, 0.0024502
7: -0.0218767, -0.0127521, -0.0215166, -0.0129696, -0.0076750, 0.0075789
8: 0.9611115, 0.9872546, 0.9621431, 0.9866316, -0.0242923, 0.0242215
9: 0.0013990, 0.0090825, 0.0015821, 0.0087793, -0.0065720, 0.0066539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150936
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0150936
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004856, 0.0009898, -0.0004822, 0.0008115, -0.0011610, 0.0013391
1: -0.0009530, 0.0028385, -0.0009372, 0.0025653, -0.0035183, 0.0037757
2: 0.0120890, 0.0177672, 0.0124981, 0.0177436, -0.0054889, 0.0051463
3: -0.0015365, 0.0027332, -0.0012288, 0.0027155, -0.0040547, 0.0037923
4: -0.0057969, -0.0018585, -0.0055131, -0.0018749, -0.0039220, 0.0036546
5: 0.0064045, 0.0106664, 0.0067116, 0.0106487, -0.0040407, 0.0037781
6: 0.0075735, 0.0105160, 0.0079878, 0.0104001, -0.0028266, 0.0025282
7: -0.0215551, -0.0123030, -0.0215166, -0.0129696, -0.0074649, 0.0081291
8: 0.9620330, 0.9885415, 0.9621431, 0.9866316, -0.0242136, 0.0257981
9: 0.0010208, 0.0088117, 0.0015821, 0.0087793, -0.0070354, 0.0065015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150936
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0150936
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005224, 0.0007949, -0.0004484, 0.0009145, -0.0012992, 0.0011322
1: -0.0011251, 0.0025399, -0.0007791, 0.0027231, -0.0038119, 0.0033190
2: 0.0125362, 0.0180250, 0.0122619, 0.0175067, -0.0049705, 0.0055589
3: -0.0012002, 0.0029271, -0.0014065, 0.0025374, -0.0036870, 0.0041172
4: -0.0054867, -0.0016797, -0.0056770, -0.0020391, -0.0034476, 0.0039973
5: 0.0067402, 0.0108599, 0.0065343, 0.0104710, -0.0036722, 0.0041038
6: 0.0080264, 0.0103893, 0.0077486, 0.0104670, -0.0024407, 0.0026408
7: -0.0219752, -0.0130317, -0.0211308, -0.0125847, -0.0083278, 0.0072168
8: 0.9608294, 0.9864537, 0.9632487, 0.9877343, -0.0260984, 0.0232049
9: 0.0016344, 0.0091654, 0.0012580, 0.0084544, -0.0062918, 0.0071827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0145762
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146014
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005219, 0.0008355, -0.0004491, 0.0009253, -0.0013039, 0.0011691
1: -0.0011231, 0.0026022, -0.0007821, 0.0027397, -0.0038389, 0.0033842
2: 0.0124430, 0.0180220, 0.0122370, 0.0175112, -0.0050683, 0.0055891
3: -0.0012703, 0.0029249, -0.0014252, 0.0025408, -0.0037853, 0.0041349
4: -0.0055514, -0.0016817, -0.0056942, -0.0020360, -0.0035154, 0.0040125
5: 0.0066702, 0.0108577, 0.0065156, 0.0104744, -0.0037697, 0.0041211
6: 0.0079319, 0.0104158, 0.0077234, 0.0104741, -0.0025422, 0.0026924
7: -0.0219703, -0.0128797, -0.0211381, -0.0125441, -0.0083116, 0.0073743
8: 0.9608434, 0.9868890, 0.9632276, 0.9878505, -0.0262526, 0.0236614
9: 0.0015065, 0.0091613, 0.0012239, 0.0084606, -0.0064377, 0.0071870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146812
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146835
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0005061, 0.0007360, -0.0010987, 0.0012789
1: -0.0010393, 0.0027206, -0.0010491, 0.0024495, -0.0034089, 0.0036660
2: 0.0122656, 0.0178965, 0.0126716, 0.0179112, -0.0053419, 0.0049526
3: -0.0014037, 0.0028305, -0.0010984, 0.0028416, -0.0039567, 0.0036596
4: -0.0056744, -0.0017688, -0.0053928, -0.0017586, -0.0039158, 0.0036240
5: 0.0065370, 0.0107635, 0.0068418, 0.0107745, -0.0039441, 0.0036470
6: 0.0077523, 0.0104660, 0.0081634, 0.0103510, -0.0025987, 0.0023026
7: -0.0217658, -0.0125907, -0.0217898, -0.0132522, -0.0072993, 0.0080094
8: 0.9614291, 0.9877172, 0.9613605, 0.9858218, -0.0232731, 0.0250835
9: 0.0012631, 0.0089891, 0.0018201, 0.0090093, -0.0069109, 0.0063382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0147766
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0147766
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0004970, 0.0007333, -0.0010864, 0.0013912
1: -0.0008953, 0.0028915, -0.0010064, 0.0024455, -0.0033408, 0.0038396
2: 0.0120096, 0.0176808, 0.0126776, 0.0178472, -0.0055990, 0.0049700
3: -0.0015962, 0.0026683, -0.0010938, 0.0027935, -0.0041501, 0.0036604
4: -0.0058520, -0.0019184, -0.0053886, -0.0018029, -0.0040490, 0.0034702
5: 0.0063449, 0.0106016, 0.0068463, 0.0107265, -0.0041371, 0.0036466
6: 0.0074931, 0.0105385, 0.0081696, 0.0103493, -0.0028562, 0.0023689
7: -0.0214144, -0.0121736, -0.0216856, -0.0132621, -0.0071962, 0.0084258
8: 0.9624361, 0.9889123, 0.9616591, 0.9857935, -0.0233574, 0.0262856
9: 0.0009118, 0.0086932, 0.0018284, 0.0089216, -0.0072627, 0.0062757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0146890
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0146927
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0004974, 0.0007755, -0.0011438, 0.0012789
1: -0.0010393, 0.0027206, -0.0010082, 0.0025101, -0.0035494, 0.0037288
2: 0.0122656, 0.0178965, 0.0125809, 0.0178498, -0.0054326, 0.0051609
3: -0.0014037, 0.0028305, -0.0011666, 0.0027954, -0.0040151, 0.0038090
4: -0.0056744, -0.0017688, -0.0054557, -0.0018011, -0.0038733, 0.0036869
5: 0.0065370, 0.0107635, 0.0067737, 0.0107285, -0.0040014, 0.0037955
6: 0.0077523, 0.0104660, 0.0080716, 0.0103767, -0.0026244, 0.0023944
7: -0.0217658, -0.0125907, -0.0216898, -0.0131044, -0.0075625, 0.0080323
8: 0.9614291, 0.9877172, 0.9616470, 0.9862452, -0.0242674, 0.0255349
9: 0.0012631, 0.0089891, 0.0016957, 0.0089251, -0.0069629, 0.0065778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0149049
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0149049
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0004877, 0.0007729, -0.0011316, 0.0013920
1: -0.0008953, 0.0028915, -0.0009629, 0.0025062, -0.0034015, 0.0038544
2: 0.0120096, 0.0176808, 0.0125867, 0.0177821, -0.0057253, 0.0050941
3: -0.0015962, 0.0026683, -0.0011622, 0.0027445, -0.0042349, 0.0038123
4: -0.0058520, -0.0019184, -0.0054517, -0.0018482, -0.0040038, 0.0035333
5: 0.0063449, 0.0106016, 0.0067781, 0.0106776, -0.0042209, 0.0037976
6: 0.0074931, 0.0105385, 0.0080775, 0.0103750, -0.0028820, 0.0024610
7: -0.0214144, -0.0121736, -0.0215793, -0.0131140, -0.0074567, 0.0085085
8: 0.9624361, 0.9889123, 0.9619636, 0.9862179, -0.0237818, 0.0269029
9: 0.0009118, 0.0086932, 0.0017037, 0.0088321, -0.0073651, 0.0065125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0148231
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0148231
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005137, 0.0008697, -0.0005054, 0.0009532, -0.0012775, 0.0011918
1: -0.0010848, 0.0026544, -0.0010458, 0.0027825, -0.0036187, 0.0035461
2: 0.0123647, 0.0179645, 0.0121729, 0.0179062, -0.0050853, 0.0052266
3: -0.0013292, 0.0028817, -0.0014734, 0.0028378, -0.0037318, 0.0038497
4: -0.0056057, -0.0017216, -0.0057387, -0.0017621, -0.0038436, 0.0040171
5: 0.0066114, 0.0108146, 0.0064674, 0.0107708, -0.0037167, 0.0038352
6: 0.0078526, 0.0104379, 0.0076584, 0.0104923, -0.0026396, 0.0027795
7: -0.0218767, -0.0127521, -0.0217816, -0.0124396, -0.0076606, 0.0073204
8: 0.9611115, 0.9872546, 0.9613839, 0.9881501, -0.0245880, 0.0239631
9: 0.0013990, 0.0090825, 0.0011358, 0.0090024, -0.0063777, 0.0066411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0151347
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0151347
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004856, 0.0009898, -0.0005054, 0.0009532, -0.0012634, 0.0013210
1: -0.0009530, 0.0028385, -0.0010458, 0.0027825, -0.0036642, 0.0037945
2: 0.0120890, 0.0177672, 0.0121729, 0.0179062, -0.0054573, 0.0052618
3: -0.0015365, 0.0027332, -0.0014734, 0.0028378, -0.0040115, 0.0038611
4: -0.0057969, -0.0018585, -0.0057387, -0.0017621, -0.0040348, 0.0038802
5: 0.0064045, 0.0106664, 0.0064674, 0.0107708, -0.0039958, 0.0038453
6: 0.0075735, 0.0105160, 0.0076584, 0.0104923, -0.0029188, 0.0028576
7: -0.0215551, -0.0123030, -0.0217816, -0.0124396, -0.0075292, 0.0079265
8: 0.9620330, 0.9885415, 0.9613839, 0.9881501, -0.0247919, 0.0256996
9: 0.0010208, 0.0088117, 0.0011358, 0.0090024, -0.0068880, 0.0065693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0151347
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0151347
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005224, 0.0007949, -0.0004740, 0.0010579, -0.0014042, 0.0011119
1: -0.0011251, 0.0025399, -0.0008986, 0.0029429, -0.0038441, 0.0034385
2: 0.0125362, 0.0180250, 0.0119327, 0.0176857, -0.0049734, 0.0055845
3: -0.0012002, 0.0029271, -0.0016540, 0.0026720, -0.0036305, 0.0041302
4: -0.0054867, -0.0016797, -0.0059053, -0.0019150, -0.0035717, 0.0042257
5: 0.0067402, 0.0108599, 0.0062871, 0.0106053, -0.0036138, 0.0041164
6: 0.0080264, 0.0103893, 0.0074151, 0.0105603, -0.0025339, 0.0029742
7: -0.0219752, -0.0130317, -0.0214224, -0.0120482, -0.0083529, 0.0069704
8: 0.9608294, 0.9864537, 0.9624131, 0.9892714, -0.0262398, 0.0234837
9: 0.0016344, 0.0091654, 0.0008063, 0.0086999, -0.0061078, 0.0071988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146659
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146852
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005219, 0.0008355, -0.0004746, 0.0010688, -0.0014127, 0.0011504
1: -0.0011231, 0.0026022, -0.0009015, 0.0029596, -0.0038665, 0.0035036
2: 0.0124430, 0.0180220, 0.0119076, 0.0176900, -0.0051357, 0.0056135
3: -0.0012703, 0.0029249, -0.0016729, 0.0026753, -0.0037439, 0.0041508
4: -0.0055514, -0.0016817, -0.0059227, -0.0019120, -0.0036394, 0.0042409
5: 0.0066702, 0.0108577, 0.0062684, 0.0106085, -0.0037262, 0.0041371
6: 0.0079319, 0.0104158, 0.0073898, 0.0105674, -0.0026355, 0.0030259
7: -0.0219703, -0.0128797, -0.0214294, -0.0120074, -0.0083804, 0.0071447
8: 0.9608434, 0.9868890, 0.9623931, 0.9893883, -0.0263812, 0.0242602
9: 0.0015065, 0.0091613, 0.0007719, 0.0087059, -0.0062661, 0.0072336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147904
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147912
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0005337, 0.0008745, -0.0011914, 0.0012607
1: -0.0010393, 0.0027206, -0.0011783, 0.0026619, -0.0034711, 0.0036516
2: 0.0122656, 0.0178965, 0.0123536, 0.0181047, -0.0052793, 0.0049923
3: -0.0014037, 0.0028305, -0.0013376, 0.0029871, -0.0038932, 0.0036680
4: -0.0056744, -0.0017688, -0.0056134, -0.0016244, -0.0040500, 0.0038446
5: 0.0065370, 0.0107635, 0.0066031, 0.0109198, -0.0038795, 0.0036538
6: 0.0077523, 0.0104660, 0.0078414, 0.0104411, -0.0026888, 0.0026246
7: -0.0217658, -0.0125907, -0.0221051, -0.0127340, -0.0072566, 0.0077629
8: 0.9614291, 0.9877172, 0.9604574, 0.9873065, -0.0235098, 0.0248327
9: 0.0012631, 0.0089891, 0.0013838, 0.0092748, -0.0067292, 0.0063016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149715
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149715
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0005239, 0.0008719, -0.0011812, 0.0013723
1: -0.0008953, 0.0028915, -0.0011325, 0.0026579, -0.0035440, 0.0038489
2: 0.0120096, 0.0176808, 0.0123595, 0.0180360, -0.0055707, 0.0050510
3: -0.0015962, 0.0026683, -0.0013331, 0.0029354, -0.0041100, 0.0036956
4: -0.0058520, -0.0019184, -0.0056093, -0.0016720, -0.0041800, 0.0036909
5: 0.0063449, 0.0106016, 0.0066075, 0.0108682, -0.0040957, 0.0036794
6: 0.0074931, 0.0105385, 0.0078474, 0.0104394, -0.0029463, 0.0026911
7: -0.0214144, -0.0121736, -0.0219932, -0.0127437, -0.0071793, 0.0082262
8: 0.9624361, 0.9889123, 0.9607778, 0.9872789, -0.0238279, 0.0261971
9: 0.0009118, 0.0086932, 0.0013919, 0.0091806, -0.0071219, 0.0062618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0148180
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0148196
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0005227, 0.0009177, -0.0012411, 0.0012586
1: -0.0010393, 0.0027206, -0.0011265, 0.0027280, -0.0036472, 0.0037367
2: 0.0122656, 0.0178965, 0.0122545, 0.0180271, -0.0053830, 0.0052424
3: -0.0014037, 0.0028305, -0.0014121, 0.0029287, -0.0039584, 0.0038511
4: -0.0056744, -0.0017688, -0.0056821, -0.0016782, -0.0039962, 0.0039133
5: 0.0065370, 0.0107635, 0.0065287, 0.0108615, -0.0039432, 0.0038356
6: 0.0077523, 0.0104660, 0.0077410, 0.0104692, -0.0027168, 0.0027250
7: -0.0217658, -0.0125907, -0.0219786, -0.0125726, -0.0075395, 0.0077875
8: 0.9614291, 0.9877172, 0.9608197, 0.9877691, -0.0246926, 0.0253426
9: 0.0012631, 0.0089891, 0.0012478, 0.0091683, -0.0067847, 0.0065696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0150970
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0150970
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0005121, 0.0009151, -0.0012330, 0.0013731
1: -0.0008953, 0.0028915, -0.0010773, 0.0027241, -0.0036194, 0.0039553
2: 0.0120096, 0.0176808, 0.0122603, 0.0179533, -0.0057102, 0.0053075
3: -0.0015962, 0.0026683, -0.0014076, 0.0028732, -0.0042021, 0.0038813
4: -0.0058520, -0.0019184, -0.0056780, -0.0017294, -0.0041226, 0.0037597
5: 0.0063449, 0.0106016, 0.0065331, 0.0108062, -0.0041863, 0.0038643
6: 0.0074931, 0.0105385, 0.0077470, 0.0104675, -0.0029744, 0.0027915
7: -0.0214144, -0.0121736, -0.0218584, -0.0125821, -0.0074840, 0.0083096
8: 0.9624361, 0.9889123, 0.9611638, 0.9877416, -0.0250375, 0.0268732
9: 0.0009118, 0.0086932, 0.0012559, 0.0090671, -0.0072258, 0.0065519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0149512
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0149512
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.12 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0151188
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147583, upper bound: 0.0151188
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0151188
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147583, upper bound: 0.0151188
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0145914
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0146205
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147066
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147098
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0147922
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147988, upper bound: 0.0147922
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147060
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147098, upper bound: 0.0147102
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0149215, upper bound: 0.0149247
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0149215, upper bound: 0.0149247
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0148407, upper bound: 0.0148433
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0155721
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147315, upper bound: 0.0155721
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0155721
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147315, upper bound: 0.0155721
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0149950
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0150401
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0152391
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147818, upper bound: 0.0152391
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0146835, upper bound: 0.0151566
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0149025, upper bound: 0.0153901
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0149025, upper bound: 0.0153901
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0147707, upper bound: 0.0153021
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0148207, upper bound: 0.0153021
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150936
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0150936
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150936
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0150936
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0145762
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146014
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146812
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146835
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0147766
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0147766
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0146890
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0146927
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0149049
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0149049
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0148231
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0148231
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0151347
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0151347
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0151347
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152235, upper bound: 0.0151347
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146659
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146852
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147904
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147912
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149715
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152535, upper bound: 0.0149715
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0148180
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0148196
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0150970
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0153881, upper bound: 0.0150970
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0149512
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.12
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0149512

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004888, 0.0007203, -0.0004821, 0.0007354, -0.0010681, 0.0010494
1: -0.0009682, 0.0024255, -0.0009368, 0.0024486, -0.0032105, 0.0031892
2: 0.0127075, 0.0177899, 0.0126729, 0.0177430, -0.0046233, 0.0046606
3: -0.0010714, 0.0027504, -0.0010974, 0.0027151, -0.0034148, 0.0034435
4: -0.0053679, -0.0018427, -0.0053919, -0.0018753, -0.0034926, 0.0035492
5: 0.0068687, 0.0106835, 0.0068428, 0.0106483, -0.0034032, 0.0034317
6: 0.0081998, 0.0103408, 0.0081648, 0.0103506, -0.0021508, 0.0021760
7: -0.0215922, -0.0133108, -0.0215157, -0.0132544, -0.0069125, 0.0068353
8: 0.9619266, 0.9856540, 0.9621460, 0.9858155, -0.0219039, 0.0217352
9: 0.0018694, 0.0088429, 0.0018219, 0.0087785, -0.0059247, 0.0059797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155061, upper bound: 0.0154672
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155061, upper bound: 0.0155632
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004894, 0.0007313, -0.0004802, 0.0007732, -0.0011014, 0.0010559
1: -0.0009709, 0.0024423, -0.0009276, 0.0025066, -0.0032943, 0.0032079
2: 0.0126823, 0.0177941, 0.0125860, 0.0177291, -0.0046460, 0.0047765
3: -0.0010903, 0.0027535, -0.0011627, 0.0027047, -0.0034275, 0.0035273
4: -0.0053853, -0.0018398, -0.0054521, -0.0018849, -0.0035005, 0.0036123
5: 0.0068498, 0.0106867, 0.0067776, 0.0106379, -0.0034155, 0.0035151
6: 0.0081743, 0.0103480, 0.0080768, 0.0103752, -0.0022009, 0.0022711
7: -0.0215990, -0.0132697, -0.0214931, -0.0131128, -0.0070608, 0.0068375
8: 0.9619073, 0.9857716, 0.9622107, 0.9862211, -0.0224541, 0.0218467
9: 0.0018349, 0.0088486, 0.0017028, 0.0087595, -0.0059295, 0.0061145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155734, upper bound: 0.0154672
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155734, upper bound: 0.0155632
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004589, 0.0008398, -0.0004821, 0.0007354, -0.0010540, 0.0011648
1: -0.0008280, 0.0026087, -0.0009368, 0.0024486, -0.0032511, 0.0035112
2: 0.0124332, 0.0175799, 0.0126729, 0.0177430, -0.0050719, 0.0046848
3: -0.0012777, 0.0025925, -0.0010974, 0.0027151, -0.0037323, 0.0034448
4: -0.0055581, -0.0019883, -0.0053919, -0.0018753, -0.0036829, 0.0034036
5: 0.0066628, 0.0105259, 0.0068428, 0.0106483, -0.0037179, 0.0034315
6: 0.0079220, 0.0104185, 0.0081648, 0.0103506, -0.0024286, 0.0022537
7: -0.0212500, -0.0128638, -0.0215157, -0.0132544, -0.0067787, 0.0073662
8: 0.9629070, 0.9869346, 0.9621460, 0.9858155, -0.0220563, 0.0238679
9: 0.0014931, 0.0085548, 0.0018219, 0.0087785, -0.0063996, 0.0059035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0150281
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0151188
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004595, 0.0008509, -0.0004802, 0.0007732, -0.0010849, 0.0011722
1: -0.0008308, 0.0026257, -0.0009276, 0.0025066, -0.0033356, 0.0035301
2: 0.0124077, 0.0175842, 0.0125860, 0.0177291, -0.0050937, 0.0047991
3: -0.0012968, 0.0025957, -0.0011627, 0.0027047, -0.0037487, 0.0035242
4: -0.0055758, -0.0019854, -0.0054521, -0.0018849, -0.0036909, 0.0034668
5: 0.0066437, 0.0105291, 0.0067776, 0.0106379, -0.0037342, 0.0035102
6: 0.0078963, 0.0104257, 0.0080768, 0.0103752, -0.0024790, 0.0023489
7: -0.0212570, -0.0128223, -0.0214931, -0.0131128, -0.0068968, 0.0073777
8: 0.9628871, 0.9870535, 0.9622107, 0.9862211, -0.0226022, 0.0239703
9: 0.0014581, 0.0085607, 0.0017028, 0.0087595, -0.0064191, 0.0060152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147560, upper bound: 0.0150281
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147560, upper bound: 0.0151188
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004967, 0.0006573, -0.0004589, 0.0008398, -0.0011779, 0.0009813
1: -0.0010051, 0.0023289, -0.0008280, 0.0026087, -0.0034927, 0.0031410
2: 0.0128522, 0.0178452, 0.0124332, 0.0175799, -0.0045155, 0.0050751
3: -0.0009626, 0.0027919, -0.0012777, 0.0025925, -0.0033174, 0.0037476
4: -0.0052675, -0.0018043, -0.0055581, -0.0019883, -0.0032792, 0.0037538
5: 0.0069773, 0.0107250, 0.0066628, 0.0105259, -0.0033043, 0.0037344
6: 0.0083463, 0.0102999, 0.0079220, 0.0104185, -0.0020722, 0.0023778
7: -0.0216823, -0.0135465, -0.0212500, -0.0128638, -0.0074938, 0.0065183
8: 0.9616687, 0.9849786, 0.9629070, 0.9869346, -0.0238521, 0.0212685
9: 0.0020679, 0.0089188, 0.0014931, 0.0085548, -0.0056756, 0.0064813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0145914
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0145914
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004967, 0.0006573, -0.0004473, 0.0008719, -0.0012132, 0.0009720
1: -0.0010051, 0.0023289, -0.0007736, 0.0026578, -0.0035440, 0.0031025
2: 0.0128522, 0.0178452, 0.0123596, 0.0174986, -0.0044718, 0.0051519
3: -0.0009626, 0.0027919, -0.0013330, 0.0025313, -0.0032816, 0.0038054
4: -0.0052675, -0.0018043, -0.0056092, -0.0020448, -0.0032228, 0.0038048
5: 0.0069773, 0.0107250, 0.0066076, 0.0104648, -0.0032684, 0.0037921
6: 0.0083463, 0.0102999, 0.0078475, 0.0104394, -0.0020930, 0.0024523
7: -0.0216823, -0.0135465, -0.0211174, -0.0127439, -0.0076189, 0.0064198
8: 0.9616687, 0.9849786, 0.9632869, 0.9872781, -0.0242107, 0.0210728
9: 0.0020679, 0.0089188, 0.0013921, 0.0084431, -0.0055947, 0.0065867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0146205
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0146205
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004949, 0.0006959, -0.0004483, 0.0008501, -0.0011892, 0.0010165
1: -0.0009967, 0.0023881, -0.0007784, 0.0026245, -0.0035251, 0.0031665
2: 0.0127636, 0.0178327, 0.0124096, 0.0175057, -0.0046044, 0.0051144
3: -0.0010292, 0.0027826, -0.0012954, 0.0025367, -0.0033804, 0.0037741
4: -0.0053290, -0.0018130, -0.0055745, -0.0020398, -0.0032892, 0.0037615
5: 0.0069108, 0.0107157, 0.0066451, 0.0104702, -0.0033670, 0.0037607
6: 0.0082566, 0.0103249, 0.0078981, 0.0104252, -0.0021686, 0.0024268
7: -0.0216620, -0.0134021, -0.0211291, -0.0128253, -0.0075377, 0.0066320
8: 0.9617269, 0.9853923, 0.9632534, 0.9870449, -0.0240452, 0.0216931
9: 0.0019463, 0.0089017, 0.0014606, 0.0084530, -0.0057783, 0.0065235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147066
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147066
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004949, 0.0006959, -0.0004469, 0.0008894, -0.0012204, 0.0010112
1: -0.0009967, 0.0023881, -0.0007719, 0.0026847, -0.0035933, 0.0031600
2: 0.0127636, 0.0178327, 0.0123193, 0.0174960, -0.0046613, 0.0052116
3: -0.0010292, 0.0027826, -0.0013633, 0.0025294, -0.0034142, 0.0038434
4: -0.0053290, -0.0018130, -0.0056371, -0.0020465, -0.0032824, 0.0038242
5: 0.0069108, 0.0107157, 0.0065774, 0.0104629, -0.0033999, 0.0038294
6: 0.0082566, 0.0103249, 0.0078067, 0.0104508, -0.0021942, 0.0025182
7: -0.0216620, -0.0134021, -0.0211133, -0.0126782, -0.0076332, 0.0066297
8: 0.9617269, 0.9853923, 0.9632989, 0.9874664, -0.0245042, 0.0219766
9: 0.0019463, 0.0089017, 0.0013368, 0.0084396, -0.0057936, 0.0066142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147098
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147098
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0004894, 0.0007313, -0.0010524, 0.0011016
1: -0.0009311, 0.0025027, -0.0009709, 0.0024423, -0.0031856, 0.0032721
2: 0.0125920, 0.0177344, 0.0126823, 0.0177941, -0.0047490, 0.0046115
3: -0.0011583, 0.0027086, -0.0010903, 0.0027535, -0.0035084, 0.0034012
4: -0.0054480, -0.0018812, -0.0053853, -0.0018398, -0.0036082, 0.0035042
5: 0.0067820, 0.0106419, 0.0068498, 0.0106867, -0.0034963, 0.0033890
6: 0.0080828, 0.0103736, 0.0081743, 0.0103480, -0.0022651, 0.0021992
7: -0.0215017, -0.0131225, -0.0215990, -0.0132697, -0.0067862, 0.0070385
8: 0.9621859, 0.9861934, 0.9619073, 0.9857716, -0.0216856, 0.0223203
9: 0.0017109, 0.0087667, 0.0018349, 0.0088486, -0.0060903, 0.0058810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147122
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147583
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0004595, 0.0008509, -0.0011683, 0.0010874
1: -0.0009311, 0.0025027, -0.0008308, 0.0026257, -0.0035045, 0.0033135
2: 0.0125920, 0.0177344, 0.0124077, 0.0175842, -0.0047745, 0.0050524
3: -0.0011583, 0.0027086, -0.0012968, 0.0025957, -0.0035106, 0.0037171
4: -0.0054480, -0.0018812, -0.0055758, -0.0019854, -0.0034627, 0.0036946
5: 0.0067820, 0.0106419, 0.0066437, 0.0105291, -0.0034969, 0.0037026
6: 0.0080828, 0.0103736, 0.0078963, 0.0104257, -0.0023429, 0.0024773
7: -0.0215017, -0.0131225, -0.0212570, -0.0128223, -0.0073184, 0.0069056
8: 0.9621859, 0.9861934, 0.9628871, 0.9870535, -0.0237839, 0.0224801
9: 0.0017109, 0.0087667, 0.0014581, 0.0085607, -0.0060149, 0.0063567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147122
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147583
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004473, 0.0008719, -0.0004967, 0.0006573, -0.0009720, 0.0012132
1: -0.0007736, 0.0026578, -0.0010051, 0.0023289, -0.0031025, 0.0035440
2: 0.0123596, 0.0174986, 0.0128522, 0.0178452, -0.0051519, 0.0044718
3: -0.0013330, 0.0025313, -0.0009626, 0.0027919, -0.0038054, 0.0032816
4: -0.0056092, -0.0020448, -0.0052675, -0.0018043, -0.0038048, 0.0032228
5: 0.0066076, 0.0104648, 0.0069773, 0.0107250, -0.0037921, 0.0032684
6: 0.0078475, 0.0104394, 0.0083463, 0.0102999, -0.0024523, 0.0020930
7: -0.0211174, -0.0127439, -0.0216823, -0.0135465, -0.0064198, 0.0076189
8: 0.9632869, 0.9872781, 0.9616687, 0.9849786, -0.0210728, 0.0242107
9: 0.0013921, 0.0084431, 0.0020679, 0.0089188, -0.0065867, 0.0055947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0146602
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147060
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0004949, 0.0006959, -0.0010082, 0.0012216
1: -0.0007766, 0.0026746, -0.0009967, 0.0023881, -0.0031647, 0.0035687
2: 0.0123345, 0.0175030, 0.0127636, 0.0178327, -0.0051798, 0.0046139
3: -0.0013519, 0.0025346, -0.0010292, 0.0027826, -0.0038233, 0.0033812
4: -0.0056266, -0.0020417, -0.0053290, -0.0018130, -0.0038136, 0.0032873
5: 0.0065887, 0.0104681, 0.0069108, 0.0107157, -0.0038098, 0.0033670
6: 0.0078221, 0.0104465, 0.0082566, 0.0103249, -0.0025029, 0.0021899
7: -0.0211247, -0.0127029, -0.0216620, -0.0134021, -0.0065824, 0.0076443
8: 0.9632663, 0.9873955, 0.9617269, 0.9853923, -0.0217511, 0.0243506
9: 0.0013576, 0.0084492, 0.0019463, 0.0089017, -0.0066133, 0.0057452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147066, upper bound: 0.0146602
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147066, upper bound: 0.0147102
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0004809, 0.0007706, -0.0011001, 0.0011001
1: -0.0009311, 0.0025027, -0.0009311, 0.0025027, -0.0033576, 0.0033576
2: 0.0125920, 0.0177344, 0.0125920, 0.0177344, -0.0048547, 0.0048547
3: -0.0011583, 0.0027086, -0.0011583, 0.0027086, -0.0035768, 0.0035768
4: -0.0054480, -0.0018812, -0.0054480, -0.0018812, -0.0035668, 0.0035668
5: 0.0067820, 0.0106419, 0.0067820, 0.0106419, -0.0035638, 0.0035638
6: 0.0080828, 0.0103736, 0.0080828, 0.0103736, -0.0022907, 0.0022907
7: -0.0215017, -0.0131225, -0.0215017, -0.0131225, -0.0070798, 0.0070798
8: 0.9621859, 0.9861934, 0.9621859, 0.9861934, -0.0228376, 0.0228376
9: 0.0017109, 0.0087667, 0.0017109, 0.0087667, -0.0061523, 0.0061523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148392
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148810
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0004479, 0.0008828, -0.0012058, 0.0010873
1: -0.0009311, 0.0025027, -0.0007766, 0.0026746, -0.0036057, 0.0032793
2: 0.0125920, 0.0177344, 0.0123345, 0.0175030, -0.0048820, 0.0052602
3: -0.0011583, 0.0027086, -0.0013519, 0.0025346, -0.0035838, 0.0038683
4: -0.0054480, -0.0018812, -0.0056266, -0.0020417, -0.0034063, 0.0037454
5: 0.0067820, 0.0106419, 0.0065887, 0.0104681, -0.0035692, 0.0038530
6: 0.0080828, 0.0103736, 0.0078221, 0.0104465, -0.0023637, 0.0025515
7: -0.0215017, -0.0131225, -0.0211247, -0.0127029, -0.0075622, 0.0069681
8: 0.9621859, 0.9861934, 0.9632663, 0.9873955, -0.0247647, 0.0229272
9: 0.0017109, 0.0087667, 0.0013576, 0.0084492, -0.0060942, 0.0065839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148392
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148810
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004473, 0.0008719, -0.0004874, 0.0006990, -0.0010224, 0.0012108
1: -0.0007736, 0.0026578, -0.0009613, 0.0023928, -0.0031664, 0.0036191
2: 0.0123596, 0.0174986, 0.0127565, 0.0177796, -0.0052700, 0.0047225
3: -0.0013330, 0.0025313, -0.0010346, 0.0027426, -0.0038869, 0.0034636
4: -0.0056092, -0.0020448, -0.0053339, -0.0018498, -0.0037594, 0.0032891
5: 0.0066076, 0.0104648, 0.0069055, 0.0106758, -0.0038726, 0.0034492
6: 0.0078475, 0.0104394, 0.0082494, 0.0103270, -0.0024794, 0.0021899
7: -0.0211174, -0.0127439, -0.0215754, -0.0133906, -0.0067300, 0.0076943
8: 0.9632869, 0.9872781, 0.9619749, 0.9854253, -0.0221384, 0.0247771
9: 0.0013921, 0.0084431, 0.0019366, 0.0088288, -0.0066780, 0.0058847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0147956
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0004856, 0.0007347, -0.0010534, 0.0012201
1: -0.0007766, 0.0026746, -0.0009531, 0.0024475, -0.0032241, 0.0036278
2: 0.0123345, 0.0175030, 0.0126745, 0.0177674, -0.0052987, 0.0048285
3: -0.0013519, 0.0025346, -0.0010962, 0.0027335, -0.0039049, 0.0035493
4: -0.0056266, -0.0020417, -0.0053908, -0.0018583, -0.0037683, 0.0033491
5: 0.0065887, 0.0104681, 0.0068440, 0.0106667, -0.0038903, 0.0035341
6: 0.0078221, 0.0104465, 0.0081664, 0.0103502, -0.0025281, 0.0022801
7: -0.0211247, -0.0127029, -0.0215556, -0.0132570, -0.0068658, 0.0077264
8: 0.9632663, 0.9873955, 0.9620317, 0.9858080, -0.0225418, 0.0249220
9: 0.0013576, 0.0084492, 0.0018242, 0.0088121, -0.0067058, 0.0060078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148407, upper bound: 0.0147956
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148407, upper bound: 0.0148433
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004888, 0.0007203, -0.0005050, 0.0008755, -0.0012233, 0.0010930
1: -0.0009682, 0.0024255, -0.0010440, 0.0026634, -0.0035135, 0.0034072
2: 0.0127075, 0.0177899, 0.0123513, 0.0179035, -0.0049634, 0.0051142
3: -0.0010714, 0.0027504, -0.0013393, 0.0028357, -0.0036739, 0.0037846
4: -0.0053679, -0.0018427, -0.0056150, -0.0017639, -0.0036039, 0.0037723
5: 0.0068687, 0.0106835, 0.0066013, 0.0107687, -0.0036617, 0.0037722
6: 0.0081998, 0.0103408, 0.0078391, 0.0104417, -0.0022419, 0.0025018
7: -0.0215922, -0.0133108, -0.0217772, -0.0127303, -0.0076517, 0.0073546
8: 0.9619266, 0.9856540, 0.9613967, 0.9873171, -0.0240218, 0.0233086
9: 0.0018694, 0.0088429, 0.0013806, 0.0089987, -0.0063812, 0.0066021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154487, upper bound: 0.0159486
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154487, upper bound: 0.0160628
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004894, 0.0007313, -0.0005034, 0.0009164, -0.0012593, 0.0010972
1: -0.0009709, 0.0024423, -0.0010365, 0.0027261, -0.0035952, 0.0034232
2: 0.0126823, 0.0177941, 0.0122573, 0.0178922, -0.0049731, 0.0052272
3: -0.0010903, 0.0027535, -0.0014099, 0.0028273, -0.0036755, 0.0038662
4: -0.0053853, -0.0018398, -0.0056801, -0.0017717, -0.0036136, 0.0038403
5: 0.0068498, 0.0106867, 0.0065308, 0.0107603, -0.0036629, 0.0038534
6: 0.0081743, 0.0103480, 0.0077439, 0.0104683, -0.0022940, 0.0026040
7: -0.0215990, -0.0132697, -0.0217589, -0.0125772, -0.0077952, 0.0073320
8: 0.9619073, 0.9857716, 0.9614490, 0.9877556, -0.0245582, 0.0233702
9: 0.0018349, 0.0088486, 0.0012518, 0.0089833, -0.0063680, 0.0067329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155149, upper bound: 0.0159486
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155149, upper bound: 0.0160628
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004589, 0.0008398, -0.0005050, 0.0008755, -0.0012091, 0.0012117
1: -0.0008280, 0.0026087, -0.0010440, 0.0026634, -0.0034913, 0.0036527
2: 0.0124332, 0.0175799, 0.0123513, 0.0179035, -0.0053635, 0.0051384
3: -0.0012777, 0.0025925, -0.0013393, 0.0028357, -0.0039617, 0.0037860
4: -0.0055581, -0.0019883, -0.0056150, -0.0017639, -0.0037942, 0.0036266
5: 0.0066628, 0.0105259, 0.0066013, 0.0107687, -0.0039480, 0.0037721
6: 0.0079220, 0.0104185, 0.0078391, 0.0104417, -0.0025197, 0.0025795
7: -0.0212500, -0.0128638, -0.0217772, -0.0127303, -0.0075179, 0.0079093
8: 0.9629070, 0.9869346, 0.9613967, 0.9873171, -0.0241742, 0.0252145
9: 0.0014931, 0.0085548, 0.0013806, 0.0089987, -0.0068570, 0.0065260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0154567
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0155721
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004595, 0.0008509, -0.0005034, 0.0009164, -0.0012428, 0.0012173
1: -0.0008308, 0.0026257, -0.0010365, 0.0027261, -0.0035569, 0.0036622
2: 0.0124077, 0.0175842, 0.0122573, 0.0178922, -0.0053893, 0.0052498
3: -0.0012968, 0.0025957, -0.0014099, 0.0028273, -0.0039773, 0.0038631
4: -0.0055758, -0.0019854, -0.0056801, -0.0017717, -0.0038041, 0.0036948
5: 0.0066437, 0.0105291, 0.0065308, 0.0107603, -0.0039633, 0.0038485
6: 0.0078963, 0.0104257, 0.0077439, 0.0104683, -0.0025721, 0.0026818
7: -0.0212570, -0.0128223, -0.0217589, -0.0125772, -0.0076312, 0.0079029
8: 0.9628871, 0.9870535, 0.9614490, 0.9877556, -0.0247063, 0.0253428
9: 0.0014581, 0.0085607, 0.0012518, 0.0089833, -0.0068656, 0.0066336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147287, upper bound: 0.0154567
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147287, upper bound: 0.0155721
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004967, 0.0006573, -0.0004850, 0.0009787, -0.0013432, 0.0010195
1: -0.0010051, 0.0023289, -0.0009502, 0.0028216, -0.0037033, 0.0032791
2: 0.0128522, 0.0178452, 0.0121143, 0.0177630, -0.0048179, 0.0054195
3: -0.0009626, 0.0027919, -0.0015174, 0.0027301, -0.0035487, 0.0040229
4: -0.0052675, -0.0018043, -0.0057793, -0.0018614, -0.0034061, 0.0039750
5: 0.0069773, 0.0107250, 0.0064235, 0.0106633, -0.0035353, 0.0040109
6: 0.0083463, 0.0102999, 0.0075991, 0.0105088, -0.0021625, 0.0027007
7: -0.0216823, -0.0135465, -0.0215482, -0.0123442, -0.0082236, 0.0069800
8: 0.9616687, 0.9849786, 0.9620526, 0.9884234, -0.0254267, 0.0226685
9: 0.0020679, 0.0089188, 0.0010555, 0.0088059, -0.0060893, 0.0070683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0149950
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0149950
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004967, 0.0006573, -0.0004727, 0.0010136, -0.0013796, 0.0010130
1: -0.0010051, 0.0023289, -0.0008925, 0.0028750, -0.0037594, 0.0032214
2: 0.0128522, 0.0178452, 0.0120343, 0.0176766, -0.0047826, 0.0055035
3: -0.0009626, 0.0027919, -0.0015776, 0.0026652, -0.0035209, 0.0040861
4: -0.0052675, -0.0018043, -0.0058348, -0.0019213, -0.0033462, 0.0040305
5: 0.0069773, 0.0107250, 0.0063634, 0.0105985, -0.0035075, 0.0040739
6: 0.0083463, 0.0102999, 0.0075181, 0.0105315, -0.0021852, 0.0027818
7: -0.0216823, -0.0135465, -0.0214075, -0.0122138, -0.0083605, 0.0069087
8: 0.9616687, 0.9849786, 0.9624557, 0.9887968, -0.0258191, 0.0225096
9: 0.0020679, 0.0089188, 0.0009457, 0.0086874, -0.0060290, 0.0071837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0150401
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0150401
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004949, 0.0006959, -0.0004736, 0.0009929, -0.0013573, 0.0010590
1: -0.0009967, 0.0023881, -0.0008971, 0.0028433, -0.0037411, 0.0032852
2: 0.0127636, 0.0178327, 0.0120819, 0.0176834, -0.0049149, 0.0054683
3: -0.0010292, 0.0027826, -0.0015418, 0.0026703, -0.0036230, 0.0040590
4: -0.0053290, -0.0018130, -0.0058018, -0.0019165, -0.0034125, 0.0039888
5: 0.0069108, 0.0107157, 0.0063992, 0.0106036, -0.0036098, 0.0040469
6: 0.0082566, 0.0103249, 0.0075663, 0.0105180, -0.0022614, 0.0027587
7: -0.0216620, -0.0134021, -0.0214187, -0.0122914, -0.0082883, 0.0071405
8: 0.9617269, 0.9853923, 0.9624238, 0.9885746, -0.0256596, 0.0229685
9: 0.0019463, 0.0089017, 0.0010110, 0.0086968, -0.0062252, 0.0071312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004949, 0.0006959, -0.0004725, 0.0010321, -0.0013893, 0.0010534
1: -0.0009967, 0.0023881, -0.0008919, 0.0029033, -0.0038131, 0.0032800
2: 0.0127636, 0.0178327, 0.0119919, 0.0176758, -0.0049122, 0.0055672
3: -0.0010292, 0.0027826, -0.0016095, 0.0026645, -0.0036486, 0.0041269
4: -0.0053290, -0.0018130, -0.0058642, -0.0019219, -0.0034071, 0.0040512
5: 0.0069108, 0.0107157, 0.0063316, 0.0105978, -0.0036346, 0.0041140
6: 0.0082566, 0.0103249, 0.0074752, 0.0105435, -0.0022869, 0.0028498
7: -0.0216620, -0.0134021, -0.0214062, -0.0121448, -0.0083939, 0.0071379
8: 0.9617269, 0.9853923, 0.9624596, 0.9889947, -0.0261311, 0.0229328
9: 0.0019463, 0.0089017, 0.0008876, 0.0086863, -0.0062385, 0.0072249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0005137, 0.0008697, -0.0012078, 0.0011408
1: -0.0009311, 0.0025027, -0.0010848, 0.0026544, -0.0034775, 0.0034853
2: 0.0125920, 0.0177344, 0.0123647, 0.0179645, -0.0050720, 0.0050486
3: -0.0011583, 0.0027086, -0.0013292, 0.0028817, -0.0037518, 0.0037299
4: -0.0054480, -0.0018812, -0.0056057, -0.0017216, -0.0037264, 0.0037245
5: 0.0067820, 0.0106419, 0.0066114, 0.0108146, -0.0037392, 0.0037171
6: 0.0080828, 0.0103736, 0.0078526, 0.0104379, -0.0023551, 0.0025209
7: -0.0215017, -0.0131225, -0.0218767, -0.0127521, -0.0074984, 0.0075118
8: 0.9621859, 0.9861934, 0.9611115, 0.9872546, -0.0237262, 0.0238247
9: 0.0017109, 0.0087667, 0.0013990, 0.0090825, -0.0065165, 0.0064807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0151712
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0152235
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0004856, 0.0009898, -0.0013336, 0.0011219
1: -0.0009311, 0.0025027, -0.0009530, 0.0028385, -0.0037030, 0.0034556
2: 0.0125920, 0.0177344, 0.0120890, 0.0177672, -0.0050461, 0.0053863
3: -0.0011583, 0.0027086, -0.0015365, 0.0027332, -0.0037170, 0.0039838
4: -0.0054480, -0.0018812, -0.0057969, -0.0018585, -0.0035896, 0.0039157
5: 0.0067820, 0.0106419, 0.0064045, 0.0106664, -0.0037029, 0.0039706
6: 0.0080828, 0.0103736, 0.0075735, 0.0105160, -0.0024332, 0.0028001
7: -0.0215017, -0.0131225, -0.0215551, -0.0123030, -0.0080487, 0.0073017
8: 0.9621859, 0.9861934, 0.9620330, 0.9885415, -0.0253028, 0.0237460
9: 0.0017109, 0.0087667, 0.0010208, 0.0088117, -0.0063641, 0.0069440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0151712
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0152235
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004473, 0.0008719, -0.0005224, 0.0007949, -0.0011271, 0.0012591
1: -0.0007736, 0.0026578, -0.0011251, 0.0025399, -0.0033135, 0.0037409
2: 0.0123596, 0.0174986, 0.0125362, 0.0180250, -0.0054525, 0.0049138
3: -0.0013330, 0.0025313, -0.0012002, 0.0029271, -0.0040372, 0.0036140
4: -0.0056092, -0.0020448, -0.0054867, -0.0016797, -0.0039295, 0.0034419
5: 0.0066076, 0.0104648, 0.0067402, 0.0108599, -0.0040240, 0.0036001
6: 0.0078475, 0.0104394, 0.0080264, 0.0103893, -0.0025418, 0.0024130
7: -0.0211174, -0.0127439, -0.0219752, -0.0130317, -0.0071400, 0.0081545
8: 0.9632869, 0.9872781, 0.9608294, 0.9864537, -0.0231363, 0.0256019
9: 0.0013921, 0.0084431, 0.0016344, 0.0091654, -0.0070367, 0.0062012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0150869
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0005219, 0.0008355, -0.0011640, 0.0012636
1: -0.0007766, 0.0026746, -0.0011231, 0.0026022, -0.0033787, 0.0037668
2: 0.0123345, 0.0175030, 0.0124430, 0.0180220, -0.0054812, 0.0050523
3: -0.0013519, 0.0025346, -0.0012703, 0.0029249, -0.0040537, 0.0037108
4: -0.0056266, -0.0020417, -0.0055514, -0.0016817, -0.0039449, 0.0035097
5: 0.0065887, 0.0104681, 0.0066702, 0.0108577, -0.0040401, 0.0036961
6: 0.0078221, 0.0104465, 0.0079319, 0.0104158, -0.0025937, 0.0025146
7: -0.0211247, -0.0127029, -0.0219703, -0.0128797, -0.0072967, 0.0081357
8: 0.9632663, 0.9873955, 0.9608434, 0.9868890, -0.0236228, 0.0257488
9: 0.0013576, 0.0084492, 0.0015065, 0.0091613, -0.0070390, 0.0063467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146812, upper bound: 0.0150869
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146812, upper bound: 0.0151566
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0005040, 0.0009129, -0.0012580, 0.0011388
1: -0.0009311, 0.0025027, -0.0010393, 0.0027206, -0.0036469, 0.0035420
2: 0.0125920, 0.0177344, 0.0122656, 0.0178965, -0.0051499, 0.0052880
3: -0.0011583, 0.0027086, -0.0014037, 0.0028305, -0.0038008, 0.0039026
4: -0.0054480, -0.0018812, -0.0056744, -0.0017688, -0.0036793, 0.0037932
5: 0.0067820, 0.0106419, 0.0065370, 0.0107635, -0.0037873, 0.0038890
6: 0.0080828, 0.0103736, 0.0077523, 0.0104660, -0.0023832, 0.0026213
7: -0.0215017, -0.0131225, -0.0217658, -0.0125907, -0.0077857, 0.0075446
8: 0.9621859, 0.9861934, 0.9614291, 0.9877172, -0.0248603, 0.0242163
9: 0.0017109, 0.0087667, 0.0012631, 0.0089891, -0.0065628, 0.0067468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153098
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153760
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0007706, -0.0004733, 0.0010244, -0.0013783, 0.0011203
1: -0.0009311, 0.0025027, -0.0008953, 0.0028915, -0.0038226, 0.0033980
2: 0.0125920, 0.0177344, 0.0120096, 0.0176808, -0.0050888, 0.0056304
3: -0.0011583, 0.0027086, -0.0015962, 0.0026683, -0.0037781, 0.0041601
4: -0.0054480, -0.0018812, -0.0058520, -0.0019184, -0.0035297, 0.0039708
5: 0.0067820, 0.0106419, 0.0063449, 0.0106016, -0.0037632, 0.0041460
6: 0.0080828, 0.0103736, 0.0074931, 0.0105385, -0.0024557, 0.0028805
7: -0.0215017, -0.0131225, -0.0214144, -0.0121736, -0.0083437, 0.0073394
8: 0.9621859, 0.9861934, 0.9624361, 0.9889123, -0.0264588, 0.0237573
9: 0.0017109, 0.0087667, 0.0009118, 0.0086932, -0.0064237, 0.0072166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153098
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153760
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004473, 0.0008719, -0.0005121, 0.0008388, -0.0011778, 0.0012550
1: -0.0007736, 0.0026578, -0.0010772, 0.0026072, -0.0033808, 0.0037351
2: 0.0123596, 0.0174986, 0.0124355, 0.0179533, -0.0055445, 0.0050631
3: -0.0013330, 0.0025313, -0.0012759, 0.0028732, -0.0040939, 0.0037894
4: -0.0056092, -0.0020448, -0.0055566, -0.0017294, -0.0038798, 0.0035118
5: 0.0066076, 0.0104648, 0.0066646, 0.0108061, -0.0040798, 0.0037743
6: 0.0078475, 0.0104394, 0.0079244, 0.0104179, -0.0025704, 0.0025150
7: -0.0211174, -0.0127439, -0.0218583, -0.0128675, -0.0074358, 0.0082023
8: 0.9632869, 0.9872781, 0.9611642, 0.9869240, -0.0236371, 0.0260628
9: 0.0013921, 0.0084431, 0.0014962, 0.0090670, -0.0070970, 0.0064791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147707, upper bound: 0.0152341
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147707, upper bound: 0.0153021
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004479, 0.0008828, -0.0005101, 0.0008781, -0.0012112, 0.0012597
1: -0.0007766, 0.0026746, -0.0010677, 0.0026674, -0.0034440, 0.0037423
2: 0.0123345, 0.0175030, 0.0123452, 0.0179390, -0.0055771, 0.0051578
3: -0.0013519, 0.0025346, -0.0013438, 0.0028625, -0.0041160, 0.0038722
4: -0.0056266, -0.0020417, -0.0056192, -0.0017393, -0.0038873, 0.0035775
5: 0.0065887, 0.0104681, 0.0065968, 0.0107954, -0.0041012, 0.0038563
6: 0.0078221, 0.0104465, 0.0078329, 0.0104434, -0.0026214, 0.0026135
7: -0.0211247, -0.0127029, -0.0218351, -0.0127204, -0.0075654, 0.0081831
8: 0.9632663, 0.9873955, 0.9612308, 0.9873453, -0.0240791, 0.0261647
9: 0.0013576, 0.0084492, 0.0013723, 0.0090475, -0.0070988, 0.0065969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148207, upper bound: 0.0152341
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148207, upper bound: 0.0153021
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005131, 0.0008585, -0.0004821, 0.0007354, -0.0011074, 0.0012047
1: -0.0010820, 0.0026373, -0.0009368, 0.0024486, -0.0034246, 0.0034821
2: 0.0123903, 0.0179604, 0.0126729, 0.0177430, -0.0050620, 0.0049841
3: -0.0013099, 0.0028786, -0.0010974, 0.0027151, -0.0037447, 0.0036870
4: -0.0055879, -0.0017244, -0.0053919, -0.0018753, -0.0037126, 0.0036675
5: 0.0066307, 0.0108115, 0.0068428, 0.0106483, -0.0037325, 0.0036747
6: 0.0078786, 0.0104307, 0.0081648, 0.0103506, -0.0024720, 0.0022659
7: -0.0218700, -0.0127939, -0.0215157, -0.0132544, -0.0073856, 0.0075501
8: 0.9611307, 0.9871348, 0.9621460, 0.9858155, -0.0234109, 0.0237832
9: 0.0014342, 0.0090768, 0.0018219, 0.0087785, -0.0065266, 0.0064056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0154376
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0155158
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005137, 0.0008697, -0.0004802, 0.0007732, -0.0011433, 0.0012113
1: -0.0010848, 0.0026544, -0.0009276, 0.0025066, -0.0035057, 0.0034998
2: 0.0123647, 0.0179645, 0.0125860, 0.0177291, -0.0050831, 0.0051007
3: -0.0013292, 0.0028817, -0.0011627, 0.0027047, -0.0037561, 0.0037739
4: -0.0056057, -0.0017216, -0.0054521, -0.0018849, -0.0037208, 0.0037306
5: 0.0066114, 0.0108146, 0.0067776, 0.0106379, -0.0037435, 0.0037612
6: 0.0078526, 0.0104379, 0.0080768, 0.0103752, -0.0025226, 0.0023611
7: -0.0218767, -0.0127521, -0.0214931, -0.0131128, -0.0075628, 0.0075497
8: 0.9611115, 0.9872546, 0.9622107, 0.9862211, -0.0239600, 0.0238873
9: 0.0013990, 0.0090825, 0.0017028, 0.0087595, -0.0065293, 0.0065616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0154376
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0155158
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004850, 0.0009787, -0.0004821, 0.0007354, -0.0010884, 0.0013307
1: -0.0009502, 0.0028216, -0.0009368, 0.0024486, -0.0033988, 0.0037071
2: 0.0121143, 0.0177630, 0.0126729, 0.0177430, -0.0053990, 0.0049573
3: -0.0015174, 0.0027301, -0.0010974, 0.0027151, -0.0039981, 0.0036514
4: -0.0057793, -0.0018614, -0.0053919, -0.0018753, -0.0039041, 0.0035305
5: 0.0064235, 0.0106633, 0.0068428, 0.0106483, -0.0039854, 0.0036378
6: 0.0075991, 0.0105088, 0.0081648, 0.0103506, -0.0027515, 0.0023441
7: -0.0215482, -0.0123442, -0.0215157, -0.0132544, -0.0071750, 0.0080992
8: 0.9620526, 0.9884234, 0.9621460, 0.9858155, -0.0233267, 0.0253564
9: 0.0010555, 0.0088059, 0.0018219, 0.0087785, -0.0069890, 0.0062525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150176
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150936
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004856, 0.0009898, -0.0004802, 0.0007732, -0.0011221, 0.0013371
1: -0.0009530, 0.0028385, -0.0009276, 0.0025066, -0.0034596, 0.0037253
2: 0.0120890, 0.0177672, 0.0125860, 0.0177291, -0.0054208, 0.0050737
3: -0.0015365, 0.0027332, -0.0011627, 0.0027047, -0.0040101, 0.0037352
4: -0.0057969, -0.0018585, -0.0054521, -0.0018849, -0.0039120, 0.0035937
5: 0.0064045, 0.0106664, 0.0067776, 0.0106379, -0.0039970, 0.0037210
6: 0.0075735, 0.0105160, 0.0080768, 0.0103752, -0.0028017, 0.0024392
7: -0.0215551, -0.0123030, -0.0214931, -0.0131128, -0.0073309, 0.0081000
8: 0.9620330, 0.9885415, 0.9622107, 0.9862211, -0.0238778, 0.0254639
9: 0.0010208, 0.0088117, 0.0017028, 0.0087595, -0.0069926, 0.0063950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0150176
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0150936
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005224, 0.0007949, -0.0004589, 0.0008398, -0.0012239, 0.0011364
1: -0.0011251, 0.0025399, -0.0008280, 0.0026087, -0.0036896, 0.0033679
2: 0.0125362, 0.0180250, 0.0124332, 0.0175799, -0.0049575, 0.0053757
3: -0.0012002, 0.0029271, -0.0012777, 0.0025925, -0.0036498, 0.0039794
4: -0.0054867, -0.0016797, -0.0055581, -0.0019883, -0.0034984, 0.0038785
5: 0.0067402, 0.0108599, 0.0066628, 0.0105259, -0.0036361, 0.0039663
6: 0.0080264, 0.0103893, 0.0079220, 0.0104185, -0.0023922, 0.0024673
7: -0.0219752, -0.0130317, -0.0212500, -0.0128638, -0.0080294, 0.0072384
8: 0.9608294, 0.9864537, 0.9629070, 0.9869346, -0.0252433, 0.0233319
9: 0.0016344, 0.0091654, 0.0014931, 0.0085548, -0.0062821, 0.0069313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0145762
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0145762
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005224, 0.0007949, -0.0004473, 0.0008719, -0.0012591, 0.0011271
1: -0.0011251, 0.0025399, -0.0007736, 0.0026578, -0.0037409, 0.0033135
2: 0.0125362, 0.0180250, 0.0123596, 0.0174986, -0.0049138, 0.0054525
3: -0.0012002, 0.0029271, -0.0013330, 0.0025313, -0.0036140, 0.0040372
4: -0.0054867, -0.0016797, -0.0056092, -0.0020448, -0.0034419, 0.0039295
5: 0.0067402, 0.0108599, 0.0066076, 0.0104648, -0.0036001, 0.0040240
6: 0.0080264, 0.0103893, 0.0078475, 0.0104394, -0.0024130, 0.0025418
7: -0.0219752, -0.0130317, -0.0211174, -0.0127439, -0.0081545, 0.0071400
8: 0.9608294, 0.9864537, 0.9632869, 0.9872781, -0.0256019, 0.0231363
9: 0.0016344, 0.0091654, 0.0013921, 0.0084431, -0.0062012, 0.0070367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146014
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146014
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005219, 0.0008355, -0.0004483, 0.0008501, -0.0012312, 0.0011712
1: -0.0011231, 0.0026022, -0.0007784, 0.0026245, -0.0037231, 0.0033806
2: 0.0124430, 0.0180220, 0.0124096, 0.0175057, -0.0050358, 0.0054158
3: -0.0012703, 0.0029249, -0.0012954, 0.0025367, -0.0037049, 0.0040045
4: -0.0055514, -0.0016817, -0.0055745, -0.0020398, -0.0035116, 0.0038928
5: 0.0066702, 0.0108577, 0.0066451, 0.0104702, -0.0036908, 0.0039910
6: 0.0079319, 0.0104158, 0.0078981, 0.0104252, -0.0024933, 0.0025176
7: -0.0219703, -0.0128797, -0.0211291, -0.0128253, -0.0080292, 0.0073351
8: 0.9608434, 0.9868890, 0.9632534, 0.9870449, -0.0254435, 0.0236356
9: 0.0015065, 0.0091613, 0.0014606, 0.0084530, -0.0063703, 0.0069492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146812
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146812
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005219, 0.0008355, -0.0004469, 0.0008894, -0.0012656, 0.0011670
1: -0.0011231, 0.0026022, -0.0007719, 0.0026847, -0.0037829, 0.0033741
2: 0.0124430, 0.0180220, 0.0123193, 0.0174960, -0.0050530, 0.0054992
3: -0.0012703, 0.0029249, -0.0013633, 0.0025294, -0.0037439, 0.0040657
4: -0.0055514, -0.0016817, -0.0056371, -0.0020465, -0.0035048, 0.0039554
5: 0.0066702, 0.0108577, 0.0065774, 0.0104629, -0.0037289, 0.0040518
6: 0.0079319, 0.0104158, 0.0078067, 0.0104508, -0.0025189, 0.0026091
7: -0.0219703, -0.0128797, -0.0211133, -0.0126782, -0.0081608, 0.0073440
8: 0.9608434, 0.9868890, 0.9632989, 0.9874664, -0.0258369, 0.0235901
9: 0.0015065, 0.0091613, 0.0013368, 0.0084396, -0.0063951, 0.0070558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146835
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146835
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0004894, 0.0007313, -0.0010938, 0.0012586
1: -0.0010393, 0.0027206, -0.0009709, 0.0024423, -0.0034017, 0.0035736
2: 0.0122656, 0.0178965, 0.0126823, 0.0177941, -0.0052005, 0.0049419
3: -0.0014037, 0.0028305, -0.0010903, 0.0027535, -0.0038478, 0.0036515
4: -0.0056744, -0.0017688, -0.0053853, -0.0018398, -0.0038346, 0.0036166
5: 0.0065370, 0.0107635, 0.0068498, 0.0106867, -0.0038352, 0.0036389
6: 0.0077523, 0.0104660, 0.0081743, 0.0103480, -0.0025957, 0.0022917
7: -0.0217658, -0.0125907, -0.0215990, -0.0132697, -0.0072817, 0.0077741
8: 0.9614291, 0.9877172, 0.9619073, 0.9857716, -0.0232227, 0.0244279
9: 0.0012631, 0.0089891, 0.0018349, 0.0088486, -0.0067097, 0.0063234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0146840
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147315
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0004595, 0.0008509, -0.0012140, 0.0012445
1: -0.0010393, 0.0027206, -0.0008308, 0.0026257, -0.0036650, 0.0035514
2: 0.0122656, 0.0178965, 0.0124077, 0.0175842, -0.0052259, 0.0053511
3: -0.0014037, 0.0028305, -0.0012968, 0.0025957, -0.0038501, 0.0039475
4: -0.0056744, -0.0017688, -0.0055758, -0.0019854, -0.0036890, 0.0038070
5: 0.0065370, 0.0107635, 0.0066437, 0.0105291, -0.0038358, 0.0039334
6: 0.0077523, 0.0104660, 0.0078963, 0.0104257, -0.0026734, 0.0025697
7: -0.0217658, -0.0125907, -0.0212570, -0.0128223, -0.0078532, 0.0076412
8: 0.9614291, 0.9877172, 0.9628871, 0.9870535, -0.0251667, 0.0245878
9: 0.0012631, 0.0089891, 0.0014581, 0.0085607, -0.0066344, 0.0068107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0146840
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147315
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004727, 0.0010136, -0.0004967, 0.0006573, -0.0010130, 0.0013796
1: -0.0008925, 0.0028750, -0.0010051, 0.0023289, -0.0032214, 0.0037594
2: 0.0120343, 0.0176766, 0.0128522, 0.0178452, -0.0055035, 0.0047826
3: -0.0015776, 0.0026652, -0.0009626, 0.0027919, -0.0040861, 0.0035209
4: -0.0058348, -0.0019213, -0.0052675, -0.0018043, -0.0040305, 0.0033462
5: 0.0063634, 0.0105985, 0.0069773, 0.0107250, -0.0040739, 0.0035075
6: 0.0075181, 0.0105315, 0.0083463, 0.0102999, -0.0027818, 0.0021852
7: -0.0214075, -0.0122138, -0.0216823, -0.0135465, -0.0069087, 0.0083605
8: 0.9624557, 0.9887968, 0.9616687, 0.9849786, -0.0225096, 0.0258191
9: 0.0009457, 0.0086874, 0.0020679, 0.0089188, -0.0071837, 0.0060290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0146460
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0146890
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0004949, 0.0006959, -0.0010501, 0.0013892
1: -0.0008953, 0.0028915, -0.0009967, 0.0023881, -0.0032834, 0.0037857
2: 0.0120096, 0.0176808, 0.0127636, 0.0178327, -0.0055351, 0.0049151
3: -0.0015962, 0.0026683, -0.0010292, 0.0027826, -0.0041093, 0.0036172
4: -0.0058520, -0.0019184, -0.0053290, -0.0018130, -0.0040390, 0.0034106
5: 0.0063449, 0.0106016, 0.0069108, 0.0107157, -0.0040971, 0.0036034
6: 0.0074931, 0.0105385, 0.0082566, 0.0103249, -0.0028319, 0.0022819
7: -0.0214144, -0.0121736, -0.0216620, -0.0134021, -0.0070875, 0.0083973
8: 0.9624361, 0.9889123, 0.9617269, 0.9853923, -0.0229563, 0.0259716
9: 0.0009118, 0.0086932, 0.0019463, 0.0089017, -0.0072229, 0.0061927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0146460
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0146927
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0004809, 0.0007706, -0.0011388, 0.0012580
1: -0.0010393, 0.0027206, -0.0009311, 0.0025027, -0.0035420, 0.0036469
2: 0.0122656, 0.0178965, 0.0125920, 0.0177344, -0.0052880, 0.0051499
3: -0.0014037, 0.0028305, -0.0011583, 0.0027086, -0.0039026, 0.0038008
4: -0.0056744, -0.0017688, -0.0054480, -0.0018812, -0.0037932, 0.0036793
5: 0.0065370, 0.0107635, 0.0067820, 0.0106419, -0.0038890, 0.0037873
6: 0.0077523, 0.0104660, 0.0080828, 0.0103736, -0.0026213, 0.0023832
7: -0.0217658, -0.0125907, -0.0215017, -0.0131225, -0.0075446, 0.0077857
8: 0.9614291, 0.9877172, 0.9621859, 0.9861934, -0.0242163, 0.0248603
9: 0.0012631, 0.0089891, 0.0017109, 0.0087667, -0.0067468, 0.0065628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148055
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148539
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0004479, 0.0008828, -0.0012498, 0.0012452
1: -0.0010393, 0.0027206, -0.0007766, 0.0026746, -0.0037139, 0.0034972
2: 0.0122656, 0.0178965, 0.0123345, 0.0175030, -0.0052374, 0.0055403
3: -0.0014037, 0.0028305, -0.0013519, 0.0025346, -0.0039096, 0.0040828
4: -0.0056744, -0.0017688, -0.0056266, -0.0020417, -0.0036327, 0.0038578
5: 0.0065370, 0.0107635, 0.0065887, 0.0104681, -0.0038944, 0.0040677
6: 0.0077523, 0.0104660, 0.0078221, 0.0104465, -0.0026942, 0.0026439
7: -0.0217658, -0.0125907, -0.0211247, -0.0127029, -0.0080755, 0.0076741
8: 0.9614291, 0.9877172, 0.9632663, 0.9873955, -0.0259663, 0.0244509
9: 0.0012631, 0.0089891, 0.0013576, 0.0084492, -0.0066887, 0.0070116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148055
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148539
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004727, 0.0010136, -0.0004874, 0.0006990, -0.0010598, 0.0013808
1: -0.0008925, 0.0028750, -0.0009613, 0.0023928, -0.0032853, 0.0038363
2: 0.0120343, 0.0176766, 0.0127565, 0.0177796, -0.0056273, 0.0049201
3: -0.0015776, 0.0026652, -0.0010346, 0.0027426, -0.0041695, 0.0036753
4: -0.0058348, -0.0019213, -0.0053339, -0.0018498, -0.0039850, 0.0034126
5: 0.0063634, 0.0105985, 0.0069055, 0.0106758, -0.0041561, 0.0036610
6: 0.0075181, 0.0105315, 0.0082494, 0.0103270, -0.0028089, 0.0022821
7: -0.0214075, -0.0122138, -0.0215754, -0.0133906, -0.0071749, 0.0084471
8: 0.9624557, 0.9887968, 0.9619749, 0.9854253, -0.0229695, 0.0264157
9: 0.0009457, 0.0086874, 0.0019366, 0.0088288, -0.0072852, 0.0062704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0147786
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0148231
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0004856, 0.0007347, -0.0010936, 0.0013901
1: -0.0008953, 0.0028915, -0.0009531, 0.0024475, -0.0033429, 0.0038447
2: 0.0120096, 0.0176808, 0.0126745, 0.0177674, -0.0056580, 0.0050063
3: -0.0015962, 0.0026683, -0.0010962, 0.0027335, -0.0041919, 0.0037566
4: -0.0058520, -0.0019184, -0.0053908, -0.0018583, -0.0039937, 0.0034724
5: 0.0063449, 0.0106016, 0.0068440, 0.0106667, -0.0041786, 0.0037417
6: 0.0074931, 0.0105385, 0.0081664, 0.0103502, -0.0028571, 0.0023721
7: -0.0214144, -0.0121736, -0.0215556, -0.0132570, -0.0073450, 0.0084796
8: 0.9624361, 0.9889123, 0.9620317, 0.9858080, -0.0233719, 0.0265701
9: 0.0009118, 0.0086932, 0.0018242, 0.0088121, -0.0073242, 0.0064218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0147786
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0148231
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005131, 0.0008585, -0.0005050, 0.0008755, -0.0012020, 0.0011833
1: -0.0010820, 0.0026373, -0.0010440, 0.0026634, -0.0034894, 0.0034692
2: 0.0123903, 0.0179604, 0.0123513, 0.0179035, -0.0049923, 0.0050374
3: -0.0013099, 0.0028786, -0.0013393, 0.0028357, -0.0036725, 0.0037089
4: -0.0055879, -0.0017244, -0.0056150, -0.0017639, -0.0038240, 0.0038905
5: 0.0066307, 0.0108115, 0.0066013, 0.0107687, -0.0036584, 0.0036948
6: 0.0078786, 0.0104307, 0.0078391, 0.0104417, -0.0025631, 0.0025916
7: -0.0218700, -0.0127939, -0.0217772, -0.0127303, -0.0073717, 0.0072904
8: 0.9611307, 0.9871348, 0.9613967, 0.9873171, -0.0237023, 0.0235076
9: 0.0014342, 0.0090768, 0.0013806, 0.0089987, -0.0063275, 0.0063934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0154398
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0155169
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005137, 0.0008697, -0.0005034, 0.0009164, -0.0012403, 0.0011899
1: -0.0010848, 0.0026544, -0.0010365, 0.0027261, -0.0035933, 0.0034871
2: 0.0123647, 0.0179645, 0.0122573, 0.0178922, -0.0050129, 0.0051722
3: -0.0013292, 0.0028817, -0.0014099, 0.0028273, -0.0036846, 0.0038032
4: -0.0056057, -0.0017216, -0.0056801, -0.0017717, -0.0038340, 0.0039585
5: 0.0066114, 0.0108146, 0.0065308, 0.0107603, -0.0036703, 0.0037886
6: 0.0078526, 0.0104379, 0.0077439, 0.0104683, -0.0026157, 0.0026940
7: -0.0218767, -0.0127521, -0.0217589, -0.0125772, -0.0075412, 0.0072923
8: 0.9611115, 0.9872546, 0.9614490, 0.9877556, -0.0243545, 0.0236066
9: 0.0013990, 0.0090825, 0.0012518, 0.0089833, -0.0063337, 0.0065419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0154398
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0155169
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004850, 0.0009787, -0.0005050, 0.0008755, -0.0011879, 0.0013127
1: -0.0009502, 0.0028216, -0.0010440, 0.0026634, -0.0035348, 0.0037169
2: 0.0121143, 0.0177630, 0.0123513, 0.0179035, -0.0053632, 0.0050718
3: -0.0015174, 0.0027301, -0.0013393, 0.0028357, -0.0039514, 0.0037197
4: -0.0057793, -0.0018614, -0.0056150, -0.0017639, -0.0040154, 0.0037536
5: 0.0064235, 0.0106633, 0.0066013, 0.0107687, -0.0039368, 0.0037043
6: 0.0075991, 0.0105088, 0.0078391, 0.0104417, -0.0028426, 0.0026698
7: -0.0215482, -0.0123442, -0.0217772, -0.0127303, -0.0072395, 0.0078948
8: 0.9620526, 0.9884234, 0.9613967, 0.9873171, -0.0239007, 0.0252392
9: 0.0010555, 0.0088059, 0.0013806, 0.0089987, -0.0068364, 0.0063206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150516
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0151347
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004856, 0.0009898, -0.0005034, 0.0009164, -0.0012234, 0.0013191
1: -0.0009530, 0.0028385, -0.0010365, 0.0027261, -0.0036310, 0.0037354
2: 0.0120890, 0.0177672, 0.0122573, 0.0178922, -0.0053848, 0.0052032
3: -0.0015365, 0.0027332, -0.0014099, 0.0028273, -0.0039643, 0.0038117
4: -0.0057969, -0.0018585, -0.0056801, -0.0017717, -0.0040252, 0.0038217
5: 0.0064045, 0.0106664, 0.0065308, 0.0107603, -0.0039495, 0.0037953
6: 0.0075735, 0.0105160, 0.0077439, 0.0104683, -0.0028948, 0.0027721
7: -0.0215551, -0.0123030, -0.0217589, -0.0125772, -0.0073753, 0.0078983
8: 0.9620330, 0.9885415, 0.9614490, 0.9877556, -0.0245272, 0.0253431
9: 0.0010208, 0.0088117, 0.0012518, 0.0089833, -0.0068441, 0.0064497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0150516
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0151348
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005224, 0.0007949, -0.0004850, 0.0009787, -0.0013268, 0.0011159
1: -0.0011251, 0.0025399, -0.0009502, 0.0028216, -0.0037156, 0.0034324
2: 0.0125362, 0.0180250, 0.0121143, 0.0177630, -0.0049035, 0.0053922
3: -0.0012002, 0.0029271, -0.0015174, 0.0027301, -0.0035921, 0.0039856
4: -0.0054867, -0.0016797, -0.0057793, -0.0018614, -0.0036253, 0.0040997
5: 0.0067402, 0.0108599, 0.0064235, 0.0106633, -0.0035769, 0.0039720
6: 0.0080264, 0.0103893, 0.0075991, 0.0105088, -0.0024825, 0.0027902
7: -0.0219752, -0.0130317, -0.0215482, -0.0123442, -0.0080395, 0.0069894
8: 0.9608294, 0.9864537, 0.9620526, 0.9884234, -0.0253421, 0.0231255
9: 0.0016344, 0.0091654, 0.0010555, 0.0088059, -0.0060987, 0.0069350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146659
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146659
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005224, 0.0007949, -0.0004727, 0.0010136, -0.0013620, 0.0011068
1: -0.0011251, 0.0025399, -0.0008925, 0.0028750, -0.0037705, 0.0034143
2: 0.0125362, 0.0180250, 0.0120343, 0.0176766, -0.0048619, 0.0054743
3: -0.0012002, 0.0029271, -0.0015776, 0.0026652, -0.0035555, 0.0040474
4: -0.0054867, -0.0016797, -0.0058348, -0.0019213, -0.0035654, 0.0041552
5: 0.0067402, 0.0108599, 0.0063634, 0.0105985, -0.0035397, 0.0040337
6: 0.0080264, 0.0103893, 0.0075181, 0.0105315, -0.0025052, 0.0028713
7: -0.0219752, -0.0130317, -0.0214075, -0.0122138, -0.0081733, 0.0068927
8: 0.9608294, 0.9864537, 0.9624557, 0.9887968, -0.0257254, 0.0229418
9: 0.0016344, 0.0091654, 0.0009457, 0.0086874, -0.0060158, 0.0070476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146852
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146852
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005219, 0.0008355, -0.0004736, 0.0009929, -0.0013392, 0.0011519
1: -0.0011231, 0.0026022, -0.0008971, 0.0028433, -0.0037457, 0.0034992
2: 0.0124430, 0.0180220, 0.0120819, 0.0176834, -0.0049984, 0.0054327
3: -0.0012703, 0.0029249, -0.0015418, 0.0026703, -0.0036556, 0.0040148
4: -0.0055514, -0.0016817, -0.0058018, -0.0019165, -0.0036348, 0.0041201
5: 0.0066702, 0.0108577, 0.0063992, 0.0106036, -0.0036395, 0.0040014
6: 0.0079319, 0.0104158, 0.0075663, 0.0105180, -0.0025861, 0.0028495
7: -0.0219703, -0.0128797, -0.0214187, -0.0122914, -0.0080857, 0.0071064
8: 0.9608434, 0.9868890, 0.9624238, 0.9885746, -0.0255370, 0.0235825
9: 0.0015065, 0.0091613, 0.0010110, 0.0086968, -0.0061964, 0.0069855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147904
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147904
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005219, 0.0008355, -0.0004725, 0.0010321, -0.0013732, 0.0011484
1: -0.0011231, 0.0026022, -0.0008919, 0.0029033, -0.0038356, 0.0034941
2: 0.0124430, 0.0180220, 0.0119919, 0.0176758, -0.0050680, 0.0055522
3: -0.0012703, 0.0029249, -0.0016095, 0.0026645, -0.0036998, 0.0040971
4: -0.0055514, -0.0016817, -0.0058642, -0.0019219, -0.0036295, 0.0041825
5: 0.0066702, 0.0108577, 0.0063316, 0.0105978, -0.0036829, 0.0040826
6: 0.0079319, 0.0104158, 0.0074752, 0.0105435, -0.0026116, 0.0029406
7: -0.0219703, -0.0128797, -0.0214062, -0.0121448, -0.0082131, 0.0071143
8: 0.9608434, 0.9868890, 0.9624596, 0.9889947, -0.0261137, 0.0239257
9: 0.0015065, 0.0091613, 0.0008876, 0.0086863, -0.0062222, 0.0070961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147912
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147912
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0005137, 0.0008697, -0.0011863, 0.0012381
1: -0.0010393, 0.0027206, -0.0010848, 0.0026544, -0.0034636, 0.0035519
2: 0.0122656, 0.0178965, 0.0123647, 0.0179645, -0.0051265, 0.0049811
3: -0.0014037, 0.0028305, -0.0013292, 0.0028817, -0.0037744, 0.0036596
4: -0.0056744, -0.0017688, -0.0056057, -0.0017216, -0.0039528, 0.0038369
5: 0.0065370, 0.0107635, 0.0066114, 0.0108146, -0.0037601, 0.0036454
6: 0.0077523, 0.0104660, 0.0078526, 0.0104379, -0.0026856, 0.0026134
7: -0.0217658, -0.0125907, -0.0218767, -0.0127521, -0.0072383, 0.0074976
8: 0.9614291, 0.9877172, 0.9611115, 0.9872546, -0.0234575, 0.0241208
9: 0.0012631, 0.0089891, 0.0013990, 0.0090825, -0.0065038, 0.0062863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147889
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0148575
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0004856, 0.0009898, -0.0013155, 0.0012239
1: -0.0010393, 0.0027206, -0.0009530, 0.0028385, -0.0037120, 0.0035974
2: 0.0122656, 0.0178965, 0.0120890, 0.0177672, -0.0051618, 0.0053530
3: -0.0014037, 0.0028305, -0.0015365, 0.0027332, -0.0037859, 0.0039393
4: -0.0056744, -0.0017688, -0.0057969, -0.0018585, -0.0038159, 0.0040281
5: 0.0065370, 0.0107635, 0.0064045, 0.0106664, -0.0037702, 0.0039245
6: 0.0077523, 0.0104660, 0.0075735, 0.0105160, -0.0027637, 0.0028925
7: -0.0217658, -0.0125907, -0.0215551, -0.0123030, -0.0078444, 0.0073662
8: 0.9614291, 0.9877172, 0.9620330, 0.9885415, -0.0251940, 0.0243247
9: 0.0012631, 0.0089891, 0.0010208, 0.0088117, -0.0064321, 0.0067966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147889
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0148575
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004727, 0.0010136, -0.0005224, 0.0007949, -0.0011068, 0.0013620
1: -0.0008925, 0.0028750, -0.0011251, 0.0025399, -0.0034143, 0.0037705
2: 0.0120343, 0.0176766, 0.0125362, 0.0180250, -0.0054743, 0.0048619
3: -0.0015776, 0.0026652, -0.0012002, 0.0029271, -0.0040474, 0.0035555
4: -0.0058348, -0.0019213, -0.0054867, -0.0016797, -0.0041552, 0.0035654
5: 0.0063634, 0.0105985, 0.0067402, 0.0108599, -0.0040337, 0.0035397
6: 0.0075181, 0.0105315, 0.0080264, 0.0103893, -0.0028713, 0.0025052
7: -0.0214075, -0.0122138, -0.0219752, -0.0130317, -0.0068927, 0.0081733
8: 0.9624557, 0.9887968, 0.9608294, 0.9864537, -0.0229418, 0.0257254
9: 0.0009457, 0.0086874, 0.0016344, 0.0091654, -0.0070476, 0.0060158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0147584
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0148180
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0005219, 0.0008355, -0.0011452, 0.0013704
1: -0.0008953, 0.0028915, -0.0011231, 0.0026022, -0.0034975, 0.0037921
2: 0.0120096, 0.0176808, 0.0124430, 0.0180220, -0.0055021, 0.0050220
3: -0.0015962, 0.0026683, -0.0012703, 0.0029249, -0.0040670, 0.0036654
4: -0.0058520, -0.0019184, -0.0055514, -0.0016817, -0.0041702, 0.0036330
5: 0.0063449, 0.0106016, 0.0066702, 0.0108577, -0.0040535, 0.0036487
6: 0.0074931, 0.0105385, 0.0079319, 0.0104158, -0.0029227, 0.0026066
7: -0.0214144, -0.0121736, -0.0219703, -0.0128797, -0.0070658, 0.0081988
8: 0.9624361, 0.9889123, 0.9608434, 0.9868890, -0.0237098, 0.0258610
9: 0.0009118, 0.0086932, 0.0015065, 0.0091613, -0.0070807, 0.0061718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0147584
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0148196
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0005040, 0.0009129, -0.0012361, 0.0012361
1: -0.0010393, 0.0027206, -0.0010393, 0.0027206, -0.0036399, 0.0036399
2: 0.0122656, 0.0178965, 0.0122656, 0.0178965, -0.0052316, 0.0052316
3: -0.0014037, 0.0028305, -0.0014037, 0.0028305, -0.0038429, 0.0038429
4: -0.0056744, -0.0017688, -0.0056744, -0.0017688, -0.0039056, 0.0039056
5: 0.0065370, 0.0107635, 0.0065370, 0.0107635, -0.0038275, 0.0038275
6: 0.0077523, 0.0104660, 0.0077523, 0.0104660, -0.0027137, 0.0027137
7: -0.0217658, -0.0125907, -0.0217658, -0.0125907, -0.0075219, 0.0075219
8: 0.9614291, 0.9877172, 0.9614291, 0.9877172, -0.0246420, 0.0246420
9: 0.0012631, 0.0089891, 0.0012631, 0.0089891, -0.0065547, 0.0065547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149014
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149735
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005040, 0.0009129, -0.0004733, 0.0010244, -0.0013591, 0.0012234
1: -0.0010393, 0.0027206, -0.0008953, 0.0028915, -0.0038935, 0.0036159
2: 0.0122656, 0.0178965, 0.0120096, 0.0176808, -0.0052600, 0.0056113
3: -0.0014037, 0.0028305, -0.0015962, 0.0026683, -0.0038517, 0.0041285
4: -0.0056744, -0.0017688, -0.0058520, -0.0019184, -0.0037560, 0.0040832
5: 0.0065370, 0.0107635, 0.0063449, 0.0106016, -0.0038353, 0.0041125
6: 0.0077523, 0.0104660, 0.0074931, 0.0105385, -0.0027862, 0.0029729
7: -0.0217658, -0.0125907, -0.0214144, -0.0121736, -0.0081406, 0.0074106
8: 0.9614291, 0.9877172, 0.9624361, 0.9889123, -0.0264148, 0.0247960
9: 0.0012631, 0.0089891, 0.0009118, 0.0086932, -0.0064983, 0.0070758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149014
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149734
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004727, 0.0010136, -0.0005121, 0.0008388, -0.0011585, 0.0013630
1: -0.0008925, 0.0028750, -0.0010772, 0.0026072, -0.0034997, 0.0038732
2: 0.0120343, 0.0176766, 0.0124355, 0.0179533, -0.0056057, 0.0051217
3: -0.0015776, 0.0026652, -0.0012759, 0.0028732, -0.0041356, 0.0037436
4: -0.0058348, -0.0019213, -0.0055566, -0.0017294, -0.0041054, 0.0036353
5: 0.0063634, 0.0105985, 0.0066646, 0.0108061, -0.0041209, 0.0037272
6: 0.0075181, 0.0105315, 0.0079244, 0.0104179, -0.0028998, 0.0026072
7: -0.0214075, -0.0122138, -0.0218583, -0.0128675, -0.0072030, 0.0082560
8: 0.9624557, 0.9887968, 0.9611642, 0.9869240, -0.0241664, 0.0263629
9: 0.0009457, 0.0086874, 0.0014962, 0.0090670, -0.0071493, 0.0063104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0148852
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0149512
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004733, 0.0010244, -0.0005101, 0.0008781, -0.0011912, 0.0013713
1: -0.0008953, 0.0028915, -0.0010677, 0.0026674, -0.0035627, 0.0038980
2: 0.0120096, 0.0176808, 0.0123452, 0.0179390, -0.0056393, 0.0052423
3: -0.0015962, 0.0026683, -0.0013438, 0.0028625, -0.0041566, 0.0038261
4: -0.0058520, -0.0019184, -0.0056192, -0.0017393, -0.0041127, 0.0037008
5: 0.0063449, 0.0106016, 0.0065968, 0.0107954, -0.0041413, 0.0038085
6: 0.0074931, 0.0105385, 0.0078329, 0.0104434, -0.0029504, 0.0027056
7: -0.0214144, -0.0121736, -0.0218351, -0.0127204, -0.0073218, 0.0082818
8: 0.9624361, 0.9889123, 0.9612308, 0.9873453, -0.0247450, 0.0265267
9: 0.0009118, 0.0086932, 0.0013723, 0.0090475, -0.0071835, 0.0064178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0148852
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0149512
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155061, upper bound: 0.0154672
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155061, upper bound: 0.0155632
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155734, upper bound: 0.0154672
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155734, upper bound: 0.0155632
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0150281
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147122, upper bound: 0.0151188
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147560, upper bound: 0.0150281
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147560, upper bound: 0.0151188
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0145914
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0145914
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0146205
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146598, upper bound: 0.0146205
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147066
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147066
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147098
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0147098
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147122
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147583
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147122
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151188, upper bound: 0.0147583
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0146602
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147060
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147066, upper bound: 0.0146602
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147066, upper bound: 0.0147102
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148392
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148810
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148392
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152265, upper bound: 0.0148810
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0147956
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0148407, upper bound: 0.0147956
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0148407, upper bound: 0.0148433
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0154487, upper bound: 0.0159486
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0154487, upper bound: 0.0160628
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155149, upper bound: 0.0159486
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155149, upper bound: 0.0160628
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0154567
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146840, upper bound: 0.0155721
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147287, upper bound: 0.0154567
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147287, upper bound: 0.0155721
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0149950
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0149950
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0150401
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146431, upper bound: 0.0150401
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0151372
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0151712
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0152235
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0151712
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150936, upper bound: 0.0152235
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0150869
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146812, upper bound: 0.0150869
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0146812, upper bound: 0.0151566
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153098
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153760
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153098
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152020, upper bound: 0.0153760
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147707, upper bound: 0.0152341
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0147707, upper bound: 0.0153021
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0148207, upper bound: 0.0152341
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0148207, upper bound: 0.0153021
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0154376
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0155158
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0154376
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0155158
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150176
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150936
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0150176
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0150936
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0145762
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0145762
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146014
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146014
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146812
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146812
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146835
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0146835
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0146840
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147315
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0146840
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147315
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0146460
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0146890
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0146460
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0146927
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148055
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148539
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148055
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0148539
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0147786
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0148231
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0147786
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0148231
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0154398
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0159826, upper bound: 0.0155169
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0154398
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0160958, upper bound: 0.0155169
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0150516
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151712, upper bound: 0.0151347
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0150516
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152233, upper bound: 0.0151348
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146659
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146659
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146852
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150955, upper bound: 0.0146852
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147904
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147904
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147912
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150869, upper bound: 0.0147912
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147889
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0148575
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0147889
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0155721, upper bound: 0.0148575
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0147584
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0150401, upper bound: 0.0148180
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0147584
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0151372, upper bound: 0.0148196
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149014
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149735
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149014
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0156860, upper bound: 0.0149734
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0148852
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0152237, upper bound: 0.0149512
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0148852
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0153006, upper bound: 0.0149512

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0004821, 0.0007354, -0.0010701, 0.0009872
1: -0.0009737, 0.0023257, -0.0009368, 0.0024486, -0.0031613, 0.0030929
2: 0.0128570, 0.0177982, 0.0126729, 0.0177430, -0.0044792, 0.0046088
3: -0.0009590, 0.0027566, -0.0010974, 0.0027151, -0.0033064, 0.0034159
4: -0.0052642, -0.0018370, -0.0053919, -0.0018753, -0.0033890, 0.0035549
5: 0.0069809, 0.0106897, 0.0068428, 0.0106483, -0.0032950, 0.0034051
6: 0.0083512, 0.0102985, 0.0081648, 0.0103506, -0.0019995, 0.0021337
7: -0.0216056, -0.0135543, -0.0215157, -0.0132544, -0.0069263, 0.0066004
8: 0.9618881, 0.9849564, 0.9621460, 0.9858155, -0.0216383, 0.0210623
9: 0.0020745, 0.0088542, 0.0018219, 0.0087785, -0.0057269, 0.0059717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0154725
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0154725
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0004821, 0.0007354, -0.0010668, 0.0010240
1: -0.0009616, 0.0023849, -0.0009368, 0.0024486, -0.0031694, 0.0031419
2: 0.0127683, 0.0177801, 0.0126729, 0.0177430, -0.0045524, 0.0046106
3: -0.0010257, 0.0027430, -0.0010974, 0.0027151, -0.0033615, 0.0034116
4: -0.0053257, -0.0018495, -0.0053919, -0.0018753, -0.0034505, 0.0035424
5: 0.0069143, 0.0106762, 0.0068428, 0.0106483, -0.0033500, 0.0034003
6: 0.0082614, 0.0103236, 0.0081648, 0.0103506, -0.0020892, 0.0021588
7: -0.0215762, -0.0134098, -0.0215157, -0.0132544, -0.0068933, 0.0067198
8: 0.9619726, 0.9853703, 0.9621460, 0.9858155, -0.0216593, 0.0214044
9: 0.0019528, 0.0088294, 0.0018219, 0.0087785, -0.0058274, 0.0059504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0155632
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0155632
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0004802, 0.0007732, -0.0011052, 0.0009831
1: -0.0009737, 0.0023257, -0.0009276, 0.0025066, -0.0032119, 0.0030936
2: 0.0128570, 0.0177982, 0.0125860, 0.0177291, -0.0044748, 0.0046846
3: -0.0009590, 0.0027566, -0.0011627, 0.0027047, -0.0032987, 0.0034729
4: -0.0052642, -0.0018370, -0.0054521, -0.0018849, -0.0033794, 0.0036152
5: 0.0069809, 0.0106897, 0.0067776, 0.0106379, -0.0032869, 0.0034621
6: 0.0083512, 0.0102985, 0.0080768, 0.0103752, -0.0020241, 0.0022217
7: -0.0216056, -0.0135543, -0.0214931, -0.0131128, -0.0070498, 0.0065585
8: 0.9618881, 0.9849564, 0.9622107, 0.9862211, -0.0219923, 0.0210472
9: 0.0020745, 0.0088542, 0.0017028, 0.0087595, -0.0056946, 0.0060758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0154672
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0154672
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0004802, 0.0007732, -0.0010995, 0.0010182
1: -0.0009616, 0.0023849, -0.0009276, 0.0025066, -0.0032462, 0.0031842
2: 0.0127683, 0.0177801, 0.0125860, 0.0177291, -0.0045963, 0.0047184
3: -0.0010257, 0.0027430, -0.0011627, 0.0027047, -0.0033860, 0.0034897
4: -0.0053257, -0.0018495, -0.0054521, -0.0018849, -0.0034409, 0.0036026
5: 0.0069143, 0.0106762, 0.0067776, 0.0106379, -0.0033736, 0.0034781
6: 0.0082614, 0.0103236, 0.0080768, 0.0103752, -0.0021139, 0.0022468
7: -0.0215762, -0.0134098, -0.0214931, -0.0131128, -0.0070319, 0.0067107
8: 0.9619726, 0.9853703, 0.9622107, 0.9862211, -0.0221695, 0.0216281
9: 0.0019528, 0.0088294, 0.0017028, 0.0087595, -0.0058349, 0.0060755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0155632
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154596, upper bound: 0.0155632
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0004821, 0.0007354, -0.0010527, 0.0011022
1: -0.0008251, 0.0025071, -0.0009368, 0.0024486, -0.0031943, 0.0034106
2: 0.0125854, 0.0175756, 0.0126729, 0.0177430, -0.0049213, 0.0046163
3: -0.0011632, 0.0025892, -0.0010974, 0.0027151, -0.0036190, 0.0034028
4: -0.0054526, -0.0019914, -0.0053919, -0.0018753, -0.0035774, 0.0034005
5: 0.0067770, 0.0105226, 0.0068428, 0.0106483, -0.0036049, 0.0033905
6: 0.0080761, 0.0103754, 0.0081648, 0.0103506, -0.0022745, 0.0022106
7: -0.0212429, -0.0131117, -0.0215157, -0.0132544, -0.0067523, 0.0071209
8: 0.9629273, 0.9862242, 0.9621460, 0.9858155, -0.0217189, 0.0231649
9: 0.0017018, 0.0085488, 0.0018219, 0.0087785, -0.0061930, 0.0058611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0150267
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0150281
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0004821, 0.0007354, -0.0010525, 0.0011401
1: -0.0008211, 0.0025707, -0.0009368, 0.0024486, -0.0032088, 0.0034723
2: 0.0124901, 0.0175697, 0.0126729, 0.0177430, -0.0050138, 0.0046321
3: -0.0012348, 0.0025848, -0.0010974, 0.0027151, -0.0036885, 0.0034102
4: -0.0055187, -0.0019954, -0.0053919, -0.0018753, -0.0036434, 0.0033965
5: 0.0067056, 0.0105183, 0.0068428, 0.0106483, -0.0036743, 0.0033973
6: 0.0079797, 0.0104024, 0.0081648, 0.0103506, -0.0023709, 0.0022376
7: -0.0212334, -0.0129566, -0.0215157, -0.0132544, -0.0067574, 0.0072715
8: 0.9629546, 0.9866688, 0.9621460, 0.9858155, -0.0217976, 0.0235966
9: 0.0015712, 0.0085408, 0.0018219, 0.0087785, -0.0063199, 0.0058713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0151169
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0151188
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0004802, 0.0007732, -0.0010878, 0.0010987
1: -0.0008251, 0.0025071, -0.0009276, 0.0025066, -0.0032449, 0.0034114
2: 0.0125854, 0.0175756, 0.0125860, 0.0177291, -0.0049160, 0.0046921
3: -0.0011632, 0.0025892, -0.0011627, 0.0027047, -0.0036150, 0.0034598
4: -0.0054526, -0.0019914, -0.0054521, -0.0018849, -0.0035677, 0.0034608
5: 0.0067770, 0.0105226, 0.0067776, 0.0106379, -0.0036008, 0.0034474
6: 0.0080761, 0.0103754, 0.0080768, 0.0103752, -0.0022991, 0.0022986
7: -0.0212429, -0.0131117, -0.0214931, -0.0131128, -0.0068759, 0.0070881
8: 0.9629273, 0.9862242, 0.9622107, 0.9862211, -0.0220729, 0.0231407
9: 0.0017018, 0.0085488, 0.0017028, 0.0087595, -0.0061753, 0.0059651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0150267
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0150281
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0004802, 0.0007732, -0.0010828, 0.0011324
1: -0.0008211, 0.0025707, -0.0009276, 0.0025066, -0.0032839, 0.0034910
2: 0.0124901, 0.0175697, 0.0125860, 0.0177291, -0.0050257, 0.0047343
3: -0.0012348, 0.0025848, -0.0011627, 0.0027047, -0.0036901, 0.0034822
4: -0.0055187, -0.0019954, -0.0054521, -0.0018849, -0.0036338, 0.0034567
5: 0.0067056, 0.0105183, 0.0067776, 0.0106379, -0.0036750, 0.0034691
6: 0.0079797, 0.0104024, 0.0080768, 0.0103752, -0.0023955, 0.0023256
7: -0.0212334, -0.0129566, -0.0214931, -0.0131128, -0.0068652, 0.0072017
8: 0.9629546, 0.9866688, 0.9622107, 0.9862211, -0.0222883, 0.0236690
9: 0.0015712, 0.0085408, 0.0017028, 0.0087595, -0.0062786, 0.0059707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0151169
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146454, upper bound: 0.0151188
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0004589, 0.0008398, -0.0011652, 0.0009731
1: -0.0009737, 0.0023257, -0.0008280, 0.0026087, -0.0034263, 0.0031182
2: 0.0128570, 0.0177982, 0.0124332, 0.0175799, -0.0044857, 0.0049726
3: -0.0009590, 0.0027566, -0.0012777, 0.0025925, -0.0032951, 0.0036705
4: -0.0052642, -0.0018370, -0.0055581, -0.0019883, -0.0032759, 0.0037212
5: 0.0069809, 0.0106897, 0.0066628, 0.0105259, -0.0032821, 0.0036575
6: 0.0083512, 0.0102985, 0.0079220, 0.0104185, -0.0020674, 0.0023765
7: -0.0216056, -0.0135543, -0.0212500, -0.0128638, -0.0073467, 0.0064543
8: 0.9618881, 0.9849564, 0.9629070, 0.9869346, -0.0233753, 0.0211268
9: 0.0020745, 0.0088542, 0.0014931, 0.0085548, -0.0056304, 0.0063480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0145914
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0145914
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0004589, 0.0008398, -0.0011203, 0.0010562
1: -0.0008251, 0.0025071, -0.0008280, 0.0026087, -0.0032925, 0.0032535
2: 0.0125854, 0.0175756, 0.0124332, 0.0175799, -0.0046540, 0.0047339
3: -0.0011632, 0.0025892, -0.0012777, 0.0025925, -0.0034041, 0.0034745
4: -0.0054526, -0.0019914, -0.0055581, -0.0019883, -0.0034643, 0.0035668
5: 0.0067770, 0.0105226, 0.0066628, 0.0105259, -0.0033891, 0.0034602
6: 0.0080761, 0.0103754, 0.0079220, 0.0104185, -0.0023424, 0.0024534
7: -0.0212429, -0.0131117, -0.0212500, -0.0128638, -0.0067634, 0.0065204
8: 0.9629273, 0.9862242, 0.9629070, 0.9869346, -0.0222974, 0.0219490
9: 0.0017018, 0.0085488, 0.0014931, 0.0085548, -0.0057049, 0.0058916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0145914
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0145914
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0004473, 0.0008719, -0.0012005, 0.0009611
1: -0.0009737, 0.0023257, -0.0007736, 0.0026578, -0.0034776, 0.0030904
2: 0.0128570, 0.0177982, 0.0123596, 0.0174986, -0.0044363, 0.0050494
3: -0.0009590, 0.0027566, -0.0013330, 0.0025313, -0.0032541, 0.0037282
4: -0.0052642, -0.0018370, -0.0056092, -0.0020448, -0.0032195, 0.0037722
5: 0.0069809, 0.0106897, 0.0066076, 0.0104648, -0.0032406, 0.0037151
6: 0.0083512, 0.0102985, 0.0078475, 0.0104394, -0.0020882, 0.0024510
7: -0.0216056, -0.0135543, -0.0211174, -0.0127439, -0.0074718, 0.0063192
8: 0.9618881, 0.9849564, 0.9632869, 0.9872781, -0.0237339, 0.0209043
9: 0.0020745, 0.0088542, 0.0013921, 0.0084431, -0.0055181, 0.0064534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0146205
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0146205
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0004473, 0.0008719, -0.0011567, 0.0010504
1: -0.0008251, 0.0025071, -0.0007736, 0.0026578, -0.0033596, 0.0032466
2: 0.0125854, 0.0175756, 0.0123596, 0.0174986, -0.0046332, 0.0048343
3: -0.0011632, 0.0025892, -0.0013330, 0.0025313, -0.0033836, 0.0035500
4: -0.0054526, -0.0019914, -0.0056092, -0.0020448, -0.0034078, 0.0036178
5: 0.0067770, 0.0105226, 0.0066076, 0.0104648, -0.0033682, 0.0035356
6: 0.0080761, 0.0103754, 0.0078475, 0.0104394, -0.0023633, 0.0025279
7: -0.0212429, -0.0131117, -0.0211174, -0.0127439, -0.0069270, 0.0064665
8: 0.9629273, 0.9862242, 0.9632869, 0.9872781, -0.0227663, 0.0218639
9: 0.0017018, 0.0085488, 0.0013921, 0.0084431, -0.0056680, 0.0060294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0146205
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146602, upper bound: 0.0146205
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0004483, 0.0008501, -0.0011742, 0.0010060
1: -0.0009616, 0.0023849, -0.0007784, 0.0026245, -0.0034623, 0.0031633
2: 0.0127683, 0.0177801, 0.0124096, 0.0175057, -0.0045543, 0.0050176
3: -0.0010257, 0.0027430, -0.0012954, 0.0025367, -0.0033448, 0.0037005
4: -0.0053257, -0.0018495, -0.0055745, -0.0020398, -0.0032859, 0.0037250
5: 0.0069143, 0.0106762, 0.0066451, 0.0104702, -0.0033314, 0.0036871
6: 0.0082614, 0.0103236, 0.0078981, 0.0104252, -0.0021638, 0.0024255
7: -0.0215762, -0.0134098, -0.0211291, -0.0128253, -0.0073550, 0.0065362
8: 0.9619726, 0.9853703, 0.9632534, 0.9870449, -0.0235933, 0.0214548
9: 0.0019528, 0.0088294, 0.0014606, 0.0084530, -0.0057030, 0.0063711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0146598
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0147066
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0004483, 0.0008501, -0.0011334, 0.0010961
1: -0.0008211, 0.0025707, -0.0007784, 0.0026245, -0.0033447, 0.0033436
2: 0.0124901, 0.0175697, 0.0124096, 0.0175057, -0.0047826, 0.0048028
3: -0.0012348, 0.0025848, -0.0012954, 0.0025367, -0.0034965, 0.0035213
4: -0.0055187, -0.0019954, -0.0055745, -0.0020398, -0.0034788, 0.0035791
5: 0.0067056, 0.0105183, 0.0066451, 0.0104702, -0.0034808, 0.0035067
6: 0.0079797, 0.0104024, 0.0078981, 0.0104252, -0.0024455, 0.0025043
7: -0.0212334, -0.0129566, -0.0211291, -0.0128253, -0.0068234, 0.0066853
8: 0.9629546, 0.9866688, 0.9632534, 0.9870449, -0.0226313, 0.0225569
9: 0.0015712, 0.0085408, 0.0014606, 0.0084530, -0.0058601, 0.0059523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0146598
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145914, upper bound: 0.0147066
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0004469, 0.0008894, -0.0012072, 0.0010004
1: -0.0009616, 0.0023849, -0.0007719, 0.0026847, -0.0035291, 0.0031568
2: 0.0127683, 0.0177801, 0.0123193, 0.0174960, -0.0046029, 0.0051099
3: -0.0010257, 0.0027430, -0.0013633, 0.0025294, -0.0033735, 0.0037659
4: -0.0053257, -0.0018495, -0.0056371, -0.0020465, -0.0032792, 0.0037876
5: 0.0069143, 0.0106762, 0.0065774, 0.0104629, -0.0033592, 0.0037519
6: 0.0082614, 0.0103236, 0.0078067, 0.0104508, -0.0021894, 0.0025169
7: -0.0215762, -0.0134098, -0.0211133, -0.0126782, -0.0074814, 0.0065305
8: 0.9619726, 0.9853703, 0.9632989, 0.9874664, -0.0240334, 0.0216971
9: 0.0019528, 0.0088294, 0.0013368, 0.0084396, -0.0057154, 0.0064793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146121, upper bound: 0.0146670
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146121, upper bound: 0.0147098
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0004469, 0.0008894, -0.0011647, 0.0010882
1: -0.0008211, 0.0025707, -0.0007719, 0.0026847, -0.0034164, 0.0033426
2: 0.0124901, 0.0175697, 0.0123193, 0.0174960, -0.0048158, 0.0049040
3: -0.0012348, 0.0025848, -0.0013633, 0.0025294, -0.0035146, 0.0035938
4: -0.0055187, -0.0019954, -0.0056371, -0.0020465, -0.0034721, 0.0036417
5: 0.0067056, 0.0105183, 0.0065774, 0.0104629, -0.0034980, 0.0035785
6: 0.0079797, 0.0104024, 0.0078067, 0.0104508, -0.0024711, 0.0025957
7: -0.0212334, -0.0129566, -0.0211133, -0.0126782, -0.0069397, 0.0066433
8: 0.9629546, 0.9866688, 0.9632989, 0.9874664, -0.0231127, 0.0227281
9: 0.0015712, 0.0085408, 0.0013368, 0.0084396, -0.0058450, 0.0060612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146121, upper bound: 0.0146670
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146121, upper bound: 0.0147098
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0004888, 0.0007203, -0.0010436, 0.0010304
1: -0.0009311, 0.0023893, -0.0009682, 0.0024255, -0.0031118, 0.0031461
2: 0.0127617, 0.0177344, 0.0127075, 0.0177899, -0.0045640, 0.0045234
3: -0.0010306, 0.0027086, -0.0010714, 0.0027504, -0.0033709, 0.0033477
4: -0.0053303, -0.0018812, -0.0053679, -0.0018427, -0.0034876, 0.0034867
5: 0.0069094, 0.0106419, 0.0068687, 0.0106835, -0.0033592, 0.0033369
6: 0.0082547, 0.0103255, 0.0081998, 0.0103408, -0.0020861, 0.0021256
7: -0.0215017, -0.0133991, -0.0215922, -0.0133108, -0.0067530, 0.0067551
8: 0.9621859, 0.9854009, 0.9619266, 0.9856540, -0.0212526, 0.0214531
9: 0.0019438, 0.0087667, 0.0018694, 0.0088429, -0.0058472, 0.0058376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154672, upper bound: 0.0155061
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154672, upper bound: 0.0155061
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0004894, 0.0007313, -0.0010504, 0.0010626
1: -0.0009217, 0.0024440, -0.0009709, 0.0024423, -0.0031303, 0.0032297
2: 0.0126798, 0.0177203, 0.0126823, 0.0177941, -0.0046797, 0.0045439
3: -0.0010922, 0.0026980, -0.0010903, 0.0027535, -0.0034546, 0.0033575
4: -0.0053871, -0.0018910, -0.0053853, -0.0018398, -0.0035473, 0.0034943
5: 0.0068479, 0.0106312, 0.0068498, 0.0106867, -0.0034424, 0.0033461
6: 0.0081718, 0.0103487, 0.0081743, 0.0103480, -0.0021762, 0.0021743
7: -0.0214787, -0.0132656, -0.0215990, -0.0132697, -0.0067570, 0.0069031
8: 0.9622519, 0.9857833, 0.9619073, 0.9857716, -0.0213560, 0.0220024
9: 0.0018314, 0.0087473, 0.0018349, 0.0088486, -0.0059818, 0.0058396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154672, upper bound: 0.0155734
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154672, upper bound: 0.0155734
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0004589, 0.0008398, -0.0011589, 0.0010162
1: -0.0009311, 0.0023893, -0.0008280, 0.0026087, -0.0034278, 0.0031866
2: 0.0127617, 0.0177344, 0.0124332, 0.0175799, -0.0045882, 0.0049620
3: -0.0010306, 0.0027086, -0.0012777, 0.0025925, -0.0033722, 0.0036573
4: -0.0053303, -0.0018812, -0.0055581, -0.0019883, -0.0033420, 0.0036769
5: 0.0069094, 0.0106419, 0.0066628, 0.0105259, -0.0033591, 0.0036438
6: 0.0082547, 0.0103255, 0.0079220, 0.0104185, -0.0021638, 0.0024034
7: -0.0215017, -0.0133991, -0.0212500, -0.0128638, -0.0072785, 0.0066214
8: 0.9621859, 0.9854009, 0.9629070, 0.9869346, -0.0233378, 0.0216055
9: 0.0019438, 0.0087667, 0.0014931, 0.0085548, -0.0057711, 0.0063043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150281, upper bound: 0.0147122
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150281, upper bound: 0.0147122
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0004595, 0.0008509, -0.0011663, 0.0010461
1: -0.0009217, 0.0024440, -0.0008308, 0.0026257, -0.0034501, 0.0032710
2: 0.0126798, 0.0177203, 0.0124077, 0.0175842, -0.0047024, 0.0049870
3: -0.0010922, 0.0026980, -0.0012968, 0.0025957, -0.0034515, 0.0036748
4: -0.0053871, -0.0018910, -0.0055758, -0.0019854, -0.0034017, 0.0036848
5: 0.0068479, 0.0106312, 0.0066437, 0.0105291, -0.0034376, 0.0036611
6: 0.0081718, 0.0103487, 0.0078963, 0.0104257, -0.0022540, 0.0024524
7: -0.0214787, -0.0132656, -0.0212570, -0.0128223, -0.0072886, 0.0067392
8: 0.9622519, 0.9857833, 0.9628871, 0.9870535, -0.0234600, 0.0221504
9: 0.0018314, 0.0087473, 0.0014581, 0.0085607, -0.0058824, 0.0063209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150281, upper bound: 0.0147560
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150281, upper bound: 0.0147583
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0008081, -0.0004967, 0.0006573, -0.0009745, 0.0011518
1: -0.0007733, 0.0025601, -0.0010051, 0.0023289, -0.0030737, 0.0034512
2: 0.0125060, 0.0174981, 0.0128522, 0.0178452, -0.0050130, 0.0044243
3: -0.0012229, 0.0025309, -0.0009626, 0.0027919, -0.0037009, 0.0032532
4: -0.0055077, -0.0020451, -0.0052675, -0.0018043, -0.0037033, 0.0032224
5: 0.0067175, 0.0104644, 0.0069773, 0.0107250, -0.0036878, 0.0032407
6: 0.0079958, 0.0103979, 0.0083463, 0.0102999, -0.0023041, 0.0020516
7: -0.0211166, -0.0129824, -0.0216823, -0.0135465, -0.0064354, 0.0073926
8: 0.9632893, 0.9865947, 0.9616687, 0.9849786, -0.0208332, 0.0235622
9: 0.0015929, 0.0084425, 0.0020679, 0.0089188, -0.0063960, 0.0055897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0146602
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0146602
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004458, 0.0008462, -0.0004967, 0.0006573, -0.0009706, 0.0011859
1: -0.0007666, 0.0026184, -0.0010051, 0.0023289, -0.0030738, 0.0035010
2: 0.0124187, 0.0174880, 0.0128522, 0.0178452, -0.0050875, 0.0044177
3: -0.0012886, 0.0025233, -0.0009626, 0.0027919, -0.0037570, 0.0032465
4: -0.0055682, -0.0020521, -0.0052675, -0.0018043, -0.0037639, 0.0032154
5: 0.0066519, 0.0104569, 0.0069773, 0.0107250, -0.0037437, 0.0032337
6: 0.0079073, 0.0104226, 0.0083463, 0.0102999, -0.0023925, 0.0020763
7: -0.0211002, -0.0128401, -0.0216823, -0.0135465, -0.0063995, 0.0075140
8: 0.9633361, 0.9870024, 0.9616687, 0.9849786, -0.0208079, 0.0239101
9: 0.0014731, 0.0084286, 0.0020679, 0.0089188, -0.0064983, 0.0055629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147060
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147060
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0008081, -0.0004949, 0.0006959, -0.0010113, 0.0011498
1: -0.0007733, 0.0025601, -0.0009967, 0.0023881, -0.0031227, 0.0034591
2: 0.0125060, 0.0174981, 0.0127636, 0.0178327, -0.0050156, 0.0044976
3: -0.0012229, 0.0025309, -0.0010292, 0.0027826, -0.0036998, 0.0033083
4: -0.0055077, -0.0020451, -0.0053290, -0.0018130, -0.0036947, 0.0032839
5: 0.0067175, 0.0104644, 0.0069108, 0.0107157, -0.0036865, 0.0032957
6: 0.0079958, 0.0103979, 0.0082566, 0.0103249, -0.0023292, 0.0021413
7: -0.0211166, -0.0129824, -0.0216620, -0.0134021, -0.0065547, 0.0073767
8: 0.9632893, 0.9865947, 0.9617269, 0.9853923, -0.0211752, 0.0235840
9: 0.0015929, 0.0084425, 0.0019463, 0.0089017, -0.0063880, 0.0056903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0146602
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0146602
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004458, 0.0008462, -0.0004949, 0.0006959, -0.0010061, 0.0011805
1: -0.0007666, 0.0026184, -0.0009967, 0.0023881, -0.0031546, 0.0035278
2: 0.0124187, 0.0174880, 0.0127636, 0.0178327, -0.0051134, 0.0045510
3: -0.0012886, 0.0025233, -0.0010292, 0.0027826, -0.0037696, 0.0033404
4: -0.0055682, -0.0020521, -0.0053290, -0.0018130, -0.0037552, 0.0032769
5: 0.0066519, 0.0104569, 0.0069108, 0.0107157, -0.0037556, 0.0033269
6: 0.0079073, 0.0104226, 0.0082566, 0.0103249, -0.0024176, 0.0021660
7: -0.0211002, -0.0128401, -0.0216620, -0.0134021, -0.0065521, 0.0074732
8: 0.9633361, 0.9870024, 0.9617269, 0.9853923, -0.0214426, 0.0240457
9: 0.0014731, 0.0084286, 0.0019463, 0.0089017, -0.0064794, 0.0057034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147102
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146205, upper bound: 0.0147102
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0004803, 0.0007599, -0.0010914, 0.0010284
1: -0.0009311, 0.0023893, -0.0009282, 0.0024863, -0.0032754, 0.0032320
2: 0.0127617, 0.0177344, 0.0126165, 0.0177301, -0.0046701, 0.0047533
3: -0.0010306, 0.0027086, -0.0011398, 0.0027054, -0.0034399, 0.0035138
4: -0.0053303, -0.0018812, -0.0054310, -0.0018842, -0.0034461, 0.0035498
5: 0.0069094, 0.0106419, 0.0068004, 0.0106387, -0.0034272, 0.0035020
6: 0.0082547, 0.0103255, 0.0081077, 0.0103666, -0.0021119, 0.0022178
7: -0.0215017, -0.0133991, -0.0214948, -0.0131625, -0.0070517, 0.0067979
8: 0.9621859, 0.9854009, 0.9622058, 0.9860787, -0.0223388, 0.0219727
9: 0.0019438, 0.0087667, 0.0017446, 0.0087609, -0.0059104, 0.0061010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154808, upper bound: 0.0155171
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154808, upper bound: 0.0155171
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0004809, 0.0007706, -0.0010983, 0.0010610
1: -0.0009217, 0.0024440, -0.0009311, 0.0025027, -0.0033013, 0.0033132
2: 0.0126798, 0.0177203, 0.0125920, 0.0177344, -0.0047820, 0.0047858
3: -0.0010922, 0.0026980, -0.0011583, 0.0027086, -0.0035206, 0.0035328
4: -0.0053871, -0.0018910, -0.0054480, -0.0018812, -0.0035059, 0.0035570
5: 0.0068479, 0.0106312, 0.0067820, 0.0106419, -0.0035074, 0.0035205
6: 0.0081718, 0.0103487, 0.0080828, 0.0103736, -0.0022018, 0.0022659
7: -0.0214787, -0.0132656, -0.0215017, -0.0131225, -0.0070522, 0.0069519
8: 0.9622519, 0.9857833, 0.9621859, 0.9861934, -0.0225005, 0.0225083
9: 0.0018314, 0.0087473, 0.0017109, 0.0087667, -0.0060466, 0.0061105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154808, upper bound: 0.0155856
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154808, upper bound: 0.0155856
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0004473, 0.0008719, -0.0011979, 0.0010155
1: -0.0009311, 0.0023893, -0.0007736, 0.0026578, -0.0035667, 0.0031629
2: 0.0127617, 0.0177344, 0.0123596, 0.0174986, -0.0046964, 0.0051670
3: -0.0010306, 0.0027086, -0.0013330, 0.0025313, -0.0034460, 0.0038068
4: -0.0053303, -0.0018812, -0.0056092, -0.0020448, -0.0032855, 0.0037280
5: 0.0069094, 0.0106419, 0.0066076, 0.0104648, -0.0034318, 0.0037925
6: 0.0082547, 0.0103255, 0.0078475, 0.0104394, -0.0021846, 0.0024780
7: -0.0215017, -0.0133991, -0.0211174, -0.0127439, -0.0075407, 0.0066858
8: 0.9621859, 0.9854009, 0.9632869, 0.9872781, -0.0243021, 0.0221140
9: 0.0019438, 0.0087667, 0.0013921, 0.0084431, -0.0058517, 0.0065373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151455, upper bound: 0.0148392
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151455, upper bound: 0.0148392
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0004479, 0.0008828, -0.0012040, 0.0010452
1: -0.0009217, 0.0024440, -0.0007766, 0.0026746, -0.0035931, 0.0032206
2: 0.0126798, 0.0177203, 0.0123345, 0.0175030, -0.0048058, 0.0051938
3: -0.0010922, 0.0026980, -0.0013519, 0.0025346, -0.0035230, 0.0038252
4: -0.0053871, -0.0018910, -0.0056266, -0.0020417, -0.0033454, 0.0037356
5: 0.0068479, 0.0106312, 0.0065887, 0.0104681, -0.0035084, 0.0038108
6: 0.0081718, 0.0103487, 0.0078221, 0.0104465, -0.0022747, 0.0025266
7: -0.0214787, -0.0132656, -0.0211247, -0.0127029, -0.0075345, 0.0067965
8: 0.9622519, 0.9857833, 0.9632663, 0.9873955, -0.0244378, 0.0225171
9: 0.0018314, 0.0087473, 0.0013576, 0.0084492, -0.0059637, 0.0065489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151455, upper bound: 0.0148810
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151455, upper bound: 0.0148810
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0008081, -0.0004874, 0.0006990, -0.0010225, 0.0011489
1: -0.0007733, 0.0025601, -0.0009613, 0.0023928, -0.0031661, 0.0035214
2: 0.0125060, 0.0174981, 0.0127565, 0.0177796, -0.0051185, 0.0046613
3: -0.0012229, 0.0025309, -0.0010346, 0.0027426, -0.0037729, 0.0034251
4: -0.0055077, -0.0020451, -0.0053339, -0.0018498, -0.0036578, 0.0032888
5: 0.0067175, 0.0104644, 0.0069055, 0.0106758, -0.0037589, 0.0034118
6: 0.0079958, 0.0103979, 0.0082494, 0.0103270, -0.0023312, 0.0021485
7: -0.0211166, -0.0129824, -0.0215754, -0.0133906, -0.0067272, 0.0074474
8: 0.9632893, 0.9865947, 0.9619749, 0.9854253, -0.0219489, 0.0240697
9: 0.0015929, 0.0084425, 0.0019366, 0.0088288, -0.0064700, 0.0058587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0147956
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0147956
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004458, 0.0008462, -0.0004874, 0.0006990, -0.0010211, 0.0011855
1: -0.0007666, 0.0026184, -0.0009613, 0.0023928, -0.0031594, 0.0035797
2: 0.0124187, 0.0174880, 0.0127565, 0.0177796, -0.0052120, 0.0046675
3: -0.0012886, 0.0025233, -0.0010346, 0.0027426, -0.0038433, 0.0034279
4: -0.0055682, -0.0020521, -0.0053339, -0.0018498, -0.0037184, 0.0032818
5: 0.0066519, 0.0104569, 0.0069055, 0.0106758, -0.0038291, 0.0034140
6: 0.0079073, 0.0104226, 0.0082494, 0.0103270, -0.0024196, 0.0021732
7: -0.0211002, -0.0128401, -0.0215754, -0.0133906, -0.0067105, 0.0075999
8: 0.9633361, 0.9870024, 0.9619749, 0.9854253, -0.0219857, 0.0245065
9: 0.0014731, 0.0084286, 0.0019366, 0.0088288, -0.0065984, 0.0058529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0008081, -0.0004856, 0.0007347, -0.0010569, 0.0011474
1: -0.0007733, 0.0025601, -0.0009531, 0.0024475, -0.0032208, 0.0035132
2: 0.0125060, 0.0174981, 0.0126745, 0.0177674, -0.0051204, 0.0047299
3: -0.0012229, 0.0025309, -0.0010962, 0.0027335, -0.0037708, 0.0034767
4: -0.0055077, -0.0020451, -0.0053908, -0.0018583, -0.0036494, 0.0033456
5: 0.0067175, 0.0104644, 0.0068440, 0.0106667, -0.0037564, 0.0034633
6: 0.0079958, 0.0103979, 0.0081664, 0.0103502, -0.0023544, 0.0022315
7: -0.0211166, -0.0129824, -0.0215556, -0.0132570, -0.0068390, 0.0074358
8: 0.9632893, 0.9865947, 0.9620317, 0.9858080, -0.0222693, 0.0240894
9: 0.0015929, 0.0084425, 0.0018242, 0.0088121, -0.0064612, 0.0059529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0147956
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0147956
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004458, 0.0008462, -0.0004856, 0.0007347, -0.0010515, 0.0011769
1: -0.0007666, 0.0026184, -0.0009531, 0.0024475, -0.0032141, 0.0035715
2: 0.0124187, 0.0174880, 0.0126745, 0.0177674, -0.0052105, 0.0047796
3: -0.0012886, 0.0025233, -0.0010962, 0.0027335, -0.0038367, 0.0035068
4: -0.0055682, -0.0020521, -0.0053908, -0.0018583, -0.0037099, 0.0033386
5: 0.0066519, 0.0104569, 0.0068440, 0.0106667, -0.0038220, 0.0034924
6: 0.0079073, 0.0104226, 0.0081664, 0.0103502, -0.0024428, 0.0022562
7: -0.0211002, -0.0128401, -0.0215556, -0.0132570, -0.0068373, 0.0075197
8: 0.9633361, 0.9870024, 0.9620317, 0.9858080, -0.0224719, 0.0245130
9: 0.0014731, 0.0084286, 0.0018242, 0.0088121, -0.0065473, 0.0059667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147891, upper bound: 0.0148433
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0005050, 0.0008755, -0.0012252, 0.0010309
1: -0.0009737, 0.0023257, -0.0010440, 0.0026634, -0.0034642, 0.0033110
2: 0.0128570, 0.0177982, 0.0123513, 0.0179035, -0.0048193, 0.0050625
3: -0.0009590, 0.0027566, -0.0013393, 0.0028357, -0.0035655, 0.0037570
4: -0.0052642, -0.0018370, -0.0056150, -0.0017639, -0.0035003, 0.0037780
5: 0.0069809, 0.0106897, 0.0066013, 0.0107687, -0.0035535, 0.0037457
6: 0.0083512, 0.0102985, 0.0078391, 0.0104417, -0.0020906, 0.0024594
7: -0.0216056, -0.0135543, -0.0217772, -0.0127303, -0.0076655, 0.0071197
8: 0.9618881, 0.9849564, 0.9613967, 0.9873171, -0.0237563, 0.0226357
9: 0.0020745, 0.0088542, 0.0013806, 0.0089987, -0.0061834, 0.0065942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0159605
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0159605
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0005050, 0.0008755, -0.0012220, 0.0010677
1: -0.0009616, 0.0023849, -0.0010440, 0.0026634, -0.0034723, 0.0033599
2: 0.0127683, 0.0177801, 0.0123513, 0.0179035, -0.0048926, 0.0050643
3: -0.0010257, 0.0027430, -0.0013393, 0.0028357, -0.0036206, 0.0037527
4: -0.0053257, -0.0018495, -0.0056150, -0.0017639, -0.0035618, 0.0037655
5: 0.0069143, 0.0106762, 0.0066013, 0.0107687, -0.0036085, 0.0037408
6: 0.0082614, 0.0103236, 0.0078391, 0.0104417, -0.0021804, 0.0024845
7: -0.0215762, -0.0134098, -0.0217772, -0.0127303, -0.0076325, 0.0072391
8: 0.9619726, 0.9853703, 0.9613967, 0.9873171, -0.0237772, 0.0229778
9: 0.0019528, 0.0088294, 0.0013806, 0.0089987, -0.0062839, 0.0065728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0160628
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0160628
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0005034, 0.0009164, -0.0012626, 0.0010244
1: -0.0009737, 0.0023257, -0.0010365, 0.0027261, -0.0035058, 0.0033088
2: 0.0128570, 0.0177982, 0.0122573, 0.0178922, -0.0048019, 0.0051249
3: -0.0009590, 0.0027566, -0.0014099, 0.0028273, -0.0035467, 0.0038039
4: -0.0052642, -0.0018370, -0.0056801, -0.0017717, -0.0034925, 0.0038432
5: 0.0069809, 0.0106897, 0.0065308, 0.0107603, -0.0035344, 0.0037925
6: 0.0083512, 0.0102985, 0.0077439, 0.0104683, -0.0021172, 0.0025546
7: -0.0216056, -0.0135543, -0.0217589, -0.0125772, -0.0077671, 0.0070529
8: 0.9618881, 0.9849564, 0.9614490, 0.9877556, -0.0240475, 0.0225707
9: 0.0020745, 0.0088542, 0.0012518, 0.0089833, -0.0061330, 0.0066798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0159486
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0159486
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0005034, 0.0009164, -0.0012573, 0.0010617
1: -0.0009616, 0.0023849, -0.0010365, 0.0027261, -0.0035471, 0.0033949
2: 0.0127683, 0.0177801, 0.0122573, 0.0178922, -0.0049260, 0.0051691
3: -0.0010257, 0.0027430, -0.0014099, 0.0028273, -0.0036405, 0.0038286
4: -0.0053257, -0.0018495, -0.0056801, -0.0017717, -0.0035540, 0.0038306
5: 0.0069143, 0.0106762, 0.0065308, 0.0107603, -0.0036279, 0.0038164
6: 0.0082614, 0.0103236, 0.0077439, 0.0104683, -0.0022070, 0.0025797
7: -0.0215762, -0.0134098, -0.0217589, -0.0125772, -0.0077663, 0.0072293
8: 0.9619726, 0.9853703, 0.9614490, 0.9877556, -0.0242736, 0.0231538
9: 0.0019528, 0.0088294, 0.0012518, 0.0089833, -0.0062911, 0.0066939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0160628
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154254, upper bound: 0.0160628
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0005050, 0.0008755, -0.0012079, 0.0011490
1: -0.0008251, 0.0025071, -0.0010440, 0.0026634, -0.0034884, 0.0035511
2: 0.0125854, 0.0175756, 0.0123513, 0.0179035, -0.0052129, 0.0050700
3: -0.0011632, 0.0025892, -0.0013393, 0.0028357, -0.0038485, 0.0037439
4: -0.0054526, -0.0019914, -0.0056150, -0.0017639, -0.0036887, 0.0036236
5: 0.0067770, 0.0105226, 0.0066013, 0.0107687, -0.0038350, 0.0037310
6: 0.0080761, 0.0103754, 0.0078391, 0.0104417, -0.0023656, 0.0025364
7: -0.0212429, -0.0131117, -0.0217772, -0.0127303, -0.0074915, 0.0076640
8: 0.9629273, 0.9862242, 0.9613967, 0.9873171, -0.0238368, 0.0245114
9: 0.0017018, 0.0085488, 0.0013806, 0.0089987, -0.0066504, 0.0064835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0154588
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0154590
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0005050, 0.0008755, -0.0012077, 0.0011869
1: -0.0008211, 0.0025707, -0.0010440, 0.0026634, -0.0034845, 0.0036146
2: 0.0124901, 0.0175697, 0.0123513, 0.0179035, -0.0053054, 0.0050857
3: -0.0012348, 0.0025848, -0.0013393, 0.0028357, -0.0039180, 0.0037513
4: -0.0055187, -0.0019954, -0.0056150, -0.0017639, -0.0037547, 0.0036196
5: 0.0067056, 0.0105183, 0.0066013, 0.0107687, -0.0039044, 0.0037378
6: 0.0079797, 0.0104024, 0.0078391, 0.0104417, -0.0024620, 0.0025633
7: -0.0212334, -0.0129566, -0.0217772, -0.0127303, -0.0074966, 0.0078146
8: 0.9629546, 0.9866688, 0.9613967, 0.9873171, -0.0239155, 0.0249431
9: 0.0015712, 0.0085408, 0.0013806, 0.0089987, -0.0067772, 0.0064937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0155670
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0155721
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0005034, 0.0009164, -0.0012453, 0.0011438
1: -0.0008251, 0.0025071, -0.0010365, 0.0027261, -0.0035389, 0.0035436
2: 0.0125854, 0.0175756, 0.0122573, 0.0178922, -0.0052116, 0.0051323
3: -0.0011632, 0.0025892, -0.0014099, 0.0028273, -0.0038437, 0.0037909
4: -0.0054526, -0.0019914, -0.0056801, -0.0017717, -0.0036809, 0.0036888
5: 0.0067770, 0.0105226, 0.0065308, 0.0107603, -0.0038299, 0.0037778
6: 0.0080761, 0.0103754, 0.0077439, 0.0104683, -0.0023922, 0.0026315
7: -0.0212429, -0.0131117, -0.0217589, -0.0125772, -0.0075932, 0.0076133
8: 0.9629273, 0.9862242, 0.9614490, 0.9877556, -0.0241281, 0.0245132
9: 0.0017018, 0.0085488, 0.0012518, 0.0089833, -0.0066217, 0.0065691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0154562
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0154567
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0005034, 0.0009164, -0.0012407, 0.0011793
1: -0.0008211, 0.0025707, -0.0010365, 0.0027261, -0.0035473, 0.0036071
2: 0.0124901, 0.0175697, 0.0122573, 0.0178922, -0.0053082, 0.0051850
3: -0.0012348, 0.0025848, -0.0014099, 0.0028273, -0.0039115, 0.0038211
4: -0.0055187, -0.0019954, -0.0056801, -0.0017717, -0.0037469, 0.0036847
5: 0.0067056, 0.0105183, 0.0065308, 0.0107603, -0.0038971, 0.0038074
6: 0.0079797, 0.0104024, 0.0077439, 0.0104683, -0.0024886, 0.0026585
7: -0.0212334, -0.0129566, -0.0217589, -0.0125772, -0.0075996, 0.0077448
8: 0.9629546, 0.9866688, 0.9614490, 0.9877556, -0.0243924, 0.0249773
9: 0.0015712, 0.0085408, 0.0012518, 0.0089833, -0.0067342, 0.0065891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0155670
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146179, upper bound: 0.0155721
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0004850, 0.0009787, -0.0013327, 0.0010075
1: -0.0009737, 0.0023257, -0.0009502, 0.0028216, -0.0036425, 0.0032759
2: 0.0128570, 0.0177982, 0.0121143, 0.0177630, -0.0047582, 0.0053295
3: -0.0009590, 0.0027566, -0.0015174, 0.0027301, -0.0035017, 0.0039578
4: -0.0052642, -0.0018370, -0.0057793, -0.0018614, -0.0034028, 0.0039424
5: 0.0069809, 0.0106897, 0.0064235, 0.0106633, -0.0034883, 0.0039461
6: 0.0083512, 0.0102985, 0.0075991, 0.0105088, -0.0021577, 0.0026994
7: -0.0216056, -0.0135543, -0.0215482, -0.0123442, -0.0081006, 0.0068506
8: 0.9618881, 0.9849564, 0.9620526, 0.9884234, -0.0250030, 0.0223972
9: 0.0020745, 0.0088542, 0.0010555, 0.0088059, -0.0059793, 0.0069606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0149950
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0149950
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0004850, 0.0009787, -0.0012941, 0.0010999
1: -0.0008251, 0.0025071, -0.0009502, 0.0028216, -0.0035552, 0.0034351
2: 0.0125854, 0.0175756, 0.0121143, 0.0177630, -0.0049285, 0.0051564
3: -0.0011632, 0.0025892, -0.0015174, 0.0027301, -0.0036169, 0.0038088
4: -0.0054526, -0.0019914, -0.0057793, -0.0018614, -0.0035912, 0.0037880
5: 0.0067770, 0.0105226, 0.0064235, 0.0106633, -0.0036024, 0.0037957
6: 0.0080761, 0.0103754, 0.0075991, 0.0105088, -0.0024327, 0.0027763
7: -0.0212429, -0.0131117, -0.0215482, -0.0123442, -0.0076532, 0.0070404
8: 0.9629273, 0.9862242, 0.9620526, 0.9884234, -0.0242387, 0.0232234
9: 0.0017018, 0.0085488, 0.0010555, 0.0088059, -0.0061473, 0.0066123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0149950
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0149950
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004900, 0.0006552, -0.0004727, 0.0010136, -0.0013690, 0.0009981
1: -0.0009737, 0.0023257, -0.0008925, 0.0028750, -0.0036986, 0.0032182
2: 0.0128570, 0.0177982, 0.0120343, 0.0176766, -0.0046954, 0.0054136
3: -0.0009590, 0.0027566, -0.0015776, 0.0026652, -0.0034553, 0.0040210
4: -0.0052642, -0.0018370, -0.0058348, -0.0019213, -0.0033429, 0.0039979
5: 0.0069809, 0.0106897, 0.0063634, 0.0105985, -0.0034421, 0.0040092
6: 0.0083512, 0.0102985, 0.0075181, 0.0105315, -0.0021803, 0.0027804
7: -0.0216056, -0.0135543, -0.0214075, -0.0122138, -0.0082376, 0.0067527
8: 0.9618881, 0.9849564, 0.9624557, 0.9887968, -0.0253953, 0.0221072
9: 0.0020745, 0.0088542, 0.0009457, 0.0086874, -0.0058990, 0.0070759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0150401
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0150401
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004583, 0.0007735, -0.0004727, 0.0010136, -0.0013318, 0.0010952
1: -0.0008251, 0.0025071, -0.0008925, 0.0028750, -0.0036255, 0.0033996
2: 0.0125854, 0.0175756, 0.0120343, 0.0176766, -0.0049070, 0.0052617
3: -0.0011632, 0.0025892, -0.0015776, 0.0026652, -0.0035976, 0.0038879
4: -0.0054526, -0.0019914, -0.0058348, -0.0019213, -0.0035313, 0.0038435
5: 0.0067770, 0.0105226, 0.0063634, 0.0105985, -0.0035826, 0.0038747
6: 0.0080761, 0.0103754, 0.0075181, 0.0105315, -0.0024554, 0.0028573
7: -0.0212429, -0.0131117, -0.0214075, -0.0122138, -0.0078247, 0.0069895
8: 0.9629273, 0.9862242, 0.9624557, 0.9887968, -0.0247301, 0.0231274
9: 0.0017018, 0.0085488, 0.0009457, 0.0086874, -0.0061075, 0.0067567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0150401
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146460, upper bound: 0.0150401
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0004736, 0.0009929, -0.0013444, 0.0010425
1: -0.0009616, 0.0023849, -0.0008971, 0.0028433, -0.0036798, 0.0032820
2: 0.0127683, 0.0177801, 0.0120819, 0.0176834, -0.0048219, 0.0053750
3: -0.0010257, 0.0027430, -0.0015418, 0.0026703, -0.0035494, 0.0039863
4: -0.0053257, -0.0018495, -0.0058018, -0.0019165, -0.0034092, 0.0039523
5: 0.0069143, 0.0106762, 0.0063992, 0.0106036, -0.0035362, 0.0039740
6: 0.0082614, 0.0103236, 0.0075663, 0.0105180, -0.0022566, 0.0027573
7: -0.0215762, -0.0134098, -0.0214187, -0.0122914, -0.0081388, 0.0069685
8: 0.9619726, 0.9853703, 0.9624238, 0.9885746, -0.0252277, 0.0226949
9: 0.0019528, 0.0088294, 0.0010110, 0.0086968, -0.0060844, 0.0069992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0150955
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0151372
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0004736, 0.0009929, -0.0013088, 0.0011425
1: -0.0008211, 0.0025707, -0.0008971, 0.0028433, -0.0036061, 0.0034677
2: 0.0124901, 0.0175697, 0.0120819, 0.0176834, -0.0050495, 0.0052245
3: -0.0012348, 0.0025848, -0.0015418, 0.0026703, -0.0037048, 0.0038563
4: -0.0055187, -0.0019954, -0.0058018, -0.0019165, -0.0036021, 0.0038064
5: 0.0067056, 0.0105183, 0.0063992, 0.0106036, -0.0036900, 0.0038427
6: 0.0079797, 0.0104024, 0.0075663, 0.0105180, -0.0025383, 0.0028361
7: -0.0212334, -0.0129566, -0.0214187, -0.0122914, -0.0077400, 0.0072207
8: 0.9629546, 0.9866688, 0.9624238, 0.9885746, -0.0245655, 0.0237940
9: 0.0015712, 0.0085408, 0.0010110, 0.0086968, -0.0063063, 0.0066897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0150955
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145762, upper bound: 0.0151372
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004874, 0.0006938, -0.0004725, 0.0010321, -0.0013785, 0.0010367
1: -0.0009616, 0.0023849, -0.0008919, 0.0029033, -0.0037529, 0.0032769
2: 0.0127683, 0.0177801, 0.0119919, 0.0176758, -0.0048654, 0.0054773
3: -0.0010257, 0.0027430, -0.0016095, 0.0026645, -0.0035741, 0.0040603
4: -0.0053257, -0.0018495, -0.0058642, -0.0019219, -0.0034039, 0.0040147
5: 0.0069143, 0.0106762, 0.0063316, 0.0105978, -0.0035600, 0.0040477
6: 0.0082614, 0.0103236, 0.0074752, 0.0105435, -0.0022821, 0.0028485
7: -0.0215762, -0.0134098, -0.0214062, -0.0121448, -0.0082684, 0.0069614
8: 0.9619726, 0.9853703, 0.9624596, 0.9889947, -0.0257123, 0.0229108
9: 0.0019528, 0.0088294, 0.0008876, 0.0086863, -0.0060922, 0.0071167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145961, upper bound: 0.0150961
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145961, upper bound: 0.0151372
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004574, 0.0008150, -0.0004725, 0.0010321, -0.0013412, 0.0011347
1: -0.0008211, 0.0025707, -0.0008919, 0.0029033, -0.0036881, 0.0034626
2: 0.0124901, 0.0175697, 0.0119919, 0.0176758, -0.0050743, 0.0053360
3: -0.0012348, 0.0025848, -0.0016095, 0.0026645, -0.0037156, 0.0039355
4: -0.0055187, -0.0019954, -0.0058642, -0.0019219, -0.0035968, 0.0038688
5: 0.0067056, 0.0105183, 0.0063316, 0.0105978, -0.0036998, 0.0039214
6: 0.0079797, 0.0104024, 0.0074752, 0.0105435, -0.0025638, 0.0029272
7: -0.0212334, -0.0129566, -0.0214062, -0.0121448, -0.0078593, 0.0071790
8: 0.9629546, 0.9866688, 0.9624596, 0.9889947, -0.0250978, 0.0239261
9: 0.0015712, 0.0085408, 0.0008876, 0.0086863, -0.0062873, 0.0068017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145961, upper bound: 0.0150961
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145961, upper bound: 0.0151372
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0005131, 0.0008585, -0.0011990, 0.0010696
1: -0.0009311, 0.0023893, -0.0010820, 0.0026373, -0.0034047, 0.0033601
2: 0.0127617, 0.0177344, 0.0123903, 0.0179604, -0.0048875, 0.0049621
3: -0.0010306, 0.0027086, -0.0013099, 0.0028786, -0.0036144, 0.0036775
4: -0.0053303, -0.0018812, -0.0055879, -0.0017244, -0.0036058, 0.0037067
5: 0.0069094, 0.0106419, 0.0066307, 0.0108115, -0.0036022, 0.0036661
6: 0.0082547, 0.0103255, 0.0078786, 0.0104307, -0.0021759, 0.0024469
7: -0.0215017, -0.0133991, -0.0218700, -0.0127939, -0.0074678, 0.0072282
8: 0.9621859, 0.9854009, 0.9611307, 0.9871348, -0.0233006, 0.0229602
9: 0.0019438, 0.0087667, 0.0014342, 0.0090768, -0.0062731, 0.0064395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154376, upper bound: 0.0159826
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154376, upper bound: 0.0159826
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0005137, 0.0008697, -0.0012058, 0.0011045
1: -0.0009217, 0.0024440, -0.0010848, 0.0026544, -0.0034221, 0.0034411
2: 0.0126798, 0.0177203, 0.0123647, 0.0179645, -0.0050040, 0.0049810
3: -0.0010922, 0.0026980, -0.0013292, 0.0028817, -0.0037011, 0.0036861
4: -0.0053871, -0.0018910, -0.0056057, -0.0017216, -0.0036655, 0.0037147
5: 0.0068479, 0.0106312, 0.0066114, 0.0108146, -0.0036886, 0.0036742
6: 0.0081718, 0.0103487, 0.0078526, 0.0104379, -0.0022662, 0.0024961
7: -0.0214787, -0.0132656, -0.0218767, -0.0127521, -0.0074692, 0.0074051
8: 0.9622519, 0.9857833, 0.9611115, 0.9872546, -0.0233966, 0.0235082
9: 0.0018314, 0.0087473, 0.0013990, 0.0090825, -0.0064289, 0.0064394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154376, upper bound: 0.0160958
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154376, upper bound: 0.0160958
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0004850, 0.0009787, -0.0013250, 0.0010506
1: -0.0009311, 0.0023893, -0.0009502, 0.0028216, -0.0036297, 0.0033395
2: 0.0127617, 0.0177344, 0.0121143, 0.0177630, -0.0048608, 0.0052991
3: -0.0010306, 0.0027086, -0.0015174, 0.0027301, -0.0035788, 0.0039309
4: -0.0053303, -0.0018812, -0.0057793, -0.0018614, -0.0034689, 0.0038981
5: 0.0069094, 0.0106419, 0.0064235, 0.0106633, -0.0035653, 0.0039191
6: 0.0082547, 0.0103255, 0.0075991, 0.0105088, -0.0022541, 0.0027264
7: -0.0215017, -0.0133991, -0.0215482, -0.0123442, -0.0080169, 0.0070177
8: 0.9621859, 0.9854009, 0.9620526, 0.9884234, -0.0248738, 0.0228759
9: 0.0019438, 0.0087667, 0.0010555, 0.0088059, -0.0061200, 0.0069019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150176, upper bound: 0.0151712
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150176, upper bound: 0.0151712
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0004856, 0.0009898, -0.0013316, 0.0010833
1: -0.0009217, 0.0024440, -0.0009530, 0.0028385, -0.0036476, 0.0033970
2: 0.0126798, 0.0177203, 0.0120890, 0.0177672, -0.0049769, 0.0053187
3: -0.0010922, 0.0026980, -0.0015365, 0.0027332, -0.0036624, 0.0039401
4: -0.0053871, -0.0018910, -0.0057969, -0.0018585, -0.0035286, 0.0039059
5: 0.0068479, 0.0106312, 0.0064045, 0.0106664, -0.0036484, 0.0039277
6: 0.0081718, 0.0103487, 0.0075735, 0.0105160, -0.0023442, 0.0027752
7: -0.0214787, -0.0132656, -0.0215551, -0.0123030, -0.0080195, 0.0071732
8: 0.9622519, 0.9857833, 0.9620330, 0.9885415, -0.0249732, 0.0234261
9: 0.0018314, 0.0087473, 0.0010208, 0.0088117, -0.0062622, 0.0069027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150176, upper bound: 0.0152233
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150176, upper bound: 0.0152235
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0008081, -0.0005224, 0.0007949, -0.0011295, 0.0011978
1: -0.0007733, 0.0025601, -0.0011251, 0.0025399, -0.0033132, 0.0036481
2: 0.0125060, 0.0174981, 0.0125362, 0.0180250, -0.0053136, 0.0048663
3: -0.0012229, 0.0025309, -0.0012002, 0.0029271, -0.0039327, 0.0035856
4: -0.0055077, -0.0020451, -0.0054867, -0.0016797, -0.0038280, 0.0034416
5: 0.0067175, 0.0104644, 0.0067402, 0.0108599, -0.0039197, 0.0035724
6: 0.0079958, 0.0103979, 0.0080264, 0.0103893, -0.0023936, 0.0023716
7: -0.0211166, -0.0129824, -0.0219752, -0.0130317, -0.0071556, 0.0079282
8: 0.9632893, 0.9865947, 0.9608294, 0.9864537, -0.0228966, 0.0249533
9: 0.0015929, 0.0084425, 0.0016344, 0.0091654, -0.0068461, 0.0061962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0150869
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0150869
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004458, 0.0008462, -0.0005224, 0.0007949, -0.0011256, 0.0012318
1: -0.0007666, 0.0026184, -0.0011251, 0.0025399, -0.0033065, 0.0036979
2: 0.0124187, 0.0174880, 0.0125362, 0.0180250, -0.0053881, 0.0048597
3: -0.0012886, 0.0025233, -0.0012002, 0.0029271, -0.0039888, 0.0035788
4: -0.0055682, -0.0020521, -0.0054867, -0.0016797, -0.0038885, 0.0034346
5: 0.0066519, 0.0104569, 0.0067402, 0.0108599, -0.0039757, 0.0035655
6: 0.0079073, 0.0104226, 0.0080264, 0.0103893, -0.0024820, 0.0023963
7: -0.0211002, -0.0128401, -0.0219752, -0.0130317, -0.0071197, 0.0080496
8: 0.9633361, 0.9870024, 0.9608294, 0.9864537, -0.0228713, 0.0253012
9: 0.0014731, 0.0084286, 0.0016344, 0.0091654, -0.0069484, 0.0061693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004472, 0.0008081, -0.0005219, 0.0008355, -0.0011660, 0.0011918
1: -0.0007733, 0.0025601, -0.0011231, 0.0026022, -0.0033754, 0.0036572
2: 0.0125060, 0.0174981, 0.0124430, 0.0180220, -0.0053170, 0.0049291
3: -0.0012229, 0.0025309, -0.0012703, 0.0029249, -0.0039302, 0.0036328
4: -0.0055077, -0.0020451, -0.0055514, -0.0016817, -0.0038259, 0.0035062
5: 0.0067175, 0.0104644, 0.0066702, 0.0108577, -0.0039168, 0.0036195
6: 0.0079958, 0.0103979, 0.0079319, 0.0104158, -0.0024200, 0.0024660
7: -0.0211166, -0.0129824, -0.0219703, -0.0128797, -0.0072578, 0.0078682
8: 0.9632893, 0.9865947, 0.9608434, 0.9868890, -0.0231896, 0.0249822
9: 0.0015929, 0.0084425, 0.0015065, 0.0091613, -0.0068137, 0.0062823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0150869
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0150869
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004458, 0.0008462, -0.0005219, 0.0008355, -0.0011619, 0.0012257
1: -0.0007666, 0.0026184, -0.0011231, 0.0026022, -0.0033687, 0.0037174
2: 0.0124187, 0.0174880, 0.0124430, 0.0180220, -0.0054010, 0.0049894
3: -0.0012886, 0.0025233, -0.0012703, 0.0029249, -0.0039918, 0.0036700
4: -0.0055682, -0.0020521, -0.0055514, -0.0016817, -0.0038865, 0.0034993
5: 0.0066519, 0.0104569, 0.0066702, 0.0108577, -0.0039781, 0.0036560
6: 0.0079073, 0.0104226, 0.0079319, 0.0104158, -0.0025084, 0.0024907
7: -0.0211002, -0.0128401, -0.0219703, -0.0128797, -0.0072664, 0.0080007
8: 0.9633361, 0.9870024, 0.9608434, 0.9868890, -0.0234892, 0.0253784
9: 0.0014731, 0.0084286, 0.0015065, 0.0091613, -0.0069210, 0.0063049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146014, upper bound: 0.0151566
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0005034, 0.0009018, -0.0012492, 0.0010670
1: -0.0009311, 0.0023893, -0.0010366, 0.0027036, -0.0035654, 0.0034259
2: 0.0127617, 0.0177344, 0.0122910, 0.0178924, -0.0049663, 0.0051876
3: -0.0010306, 0.0027086, -0.0013846, 0.0028274, -0.0036642, 0.0038404
4: -0.0053303, -0.0018812, -0.0056568, -0.0017716, -0.0035587, 0.0037756
5: 0.0069094, 0.0106419, 0.0065561, 0.0107604, -0.0036510, 0.0038280
6: 0.0082547, 0.0103255, 0.0077781, 0.0104588, -0.0022041, 0.0025474
7: -0.0215017, -0.0133991, -0.0217591, -0.0126321, -0.0077594, 0.0072627
8: 0.9621859, 0.9854009, 0.9614483, 0.9875984, -0.0243664, 0.0233544
9: 0.0019438, 0.0087667, 0.0012980, 0.0089835, -0.0063210, 0.0066969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154480, upper bound: 0.0159936
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154480, upper bound: 0.0159936
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0005040, 0.0009129, -0.0012561, 0.0011017
1: -0.0009217, 0.0024440, -0.0010393, 0.0027206, -0.0035906, 0.0034833
2: 0.0126798, 0.0177203, 0.0122656, 0.0178965, -0.0050757, 0.0052190
3: -0.0010922, 0.0026980, -0.0014037, 0.0028305, -0.0037449, 0.0038586
4: -0.0053871, -0.0018910, -0.0056744, -0.0017688, -0.0036183, 0.0037834
5: 0.0068479, 0.0106312, 0.0065370, 0.0107635, -0.0037317, 0.0038457
6: 0.0081718, 0.0103487, 0.0077523, 0.0104660, -0.0022942, 0.0025964
7: -0.0214787, -0.0132656, -0.0217658, -0.0125907, -0.0077582, 0.0074353
8: 0.9622519, 0.9857833, 0.9614291, 0.9877172, -0.0245232, 0.0238692
9: 0.0018314, 0.0087473, 0.0012631, 0.0089891, -0.0064690, 0.0067050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154480, upper bound: 0.0161130
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154480, upper bound: 0.0161130
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004809, 0.0006967, -0.0004727, 0.0010136, -0.0013698, 0.0010485
1: -0.0009311, 0.0023893, -0.0008925, 0.0028750, -0.0037935, 0.0032818
2: 0.0127617, 0.0177344, 0.0120343, 0.0176766, -0.0049149, 0.0055293
3: -0.0010306, 0.0027086, -0.0015776, 0.0026652, -0.0036408, 0.0040972
4: -0.0053303, -0.0018812, -0.0058348, -0.0019213, -0.0034090, 0.0039536
5: 0.0069094, 0.0106419, 0.0063634, 0.0105985, -0.0036262, 0.0040844
6: 0.0082547, 0.0103255, 0.0075181, 0.0105315, -0.0022768, 0.0028074
7: -0.0215017, -0.0133991, -0.0214075, -0.0122138, -0.0083160, 0.0070573
8: 0.9621859, 0.9854009, 0.9624557, 0.9887968, -0.0259613, 0.0229452
9: 0.0019438, 0.0087667, 0.0009457, 0.0086874, -0.0061810, 0.0071657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151329, upper bound: 0.0153098
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151329, upper bound: 0.0153098
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004789, 0.0007324, -0.0004733, 0.0010244, -0.0013765, 0.0010815
1: -0.0009217, 0.0024440, -0.0008953, 0.0028915, -0.0038132, 0.0033393
2: 0.0126798, 0.0177203, 0.0120096, 0.0176808, -0.0050010, 0.0055614
3: -0.0010922, 0.0026980, -0.0015962, 0.0026683, -0.0037193, 0.0041161
4: -0.0053871, -0.0018910, -0.0058520, -0.0019184, -0.0034687, 0.0039610
5: 0.0068479, 0.0106312, 0.0063449, 0.0106016, -0.0037044, 0.0041027
6: 0.0081718, 0.0103487, 0.0074931, 0.0105385, -0.0023667, 0.0028556
7: -0.0214787, -0.0132656, -0.0214144, -0.0121736, -0.0083161, 0.0072126
8: 0.9622519, 0.9857833, 0.9624361, 0.9889123, -0.0261217, 0.0233473
9: 0.0018314, 0.0087473, 0.0009118, 0.0086932, -0.0063189, 0.0071748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.16 + 597.51 = 600.66 seconds
