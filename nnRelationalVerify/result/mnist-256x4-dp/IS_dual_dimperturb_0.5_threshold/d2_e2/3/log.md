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
0: (-0.0005476, 0.0007758, -0.0005476, 0.0007758, -0.0011737, 0.0011737)
1: (-0.0012431, 0.0025106, -0.0012431, 0.0025106, -0.0037016, 0.0037016)
2: (0.0125801, 0.0182017, 0.0125801, 0.0182017, -0.0053634, 0.0053634)
3: (-0.0011672, 0.0030600, -0.0011672, 0.0030600, -0.0039574, 0.0039574)
4: (-0.0054562, -0.0015571, -0.0054562, -0.0015571, -0.0038992, 0.0038992)
5: (0.0067731, 0.0109926, 0.0067731, 0.0109926, -0.0039432, 0.0039432)
6: (0.0080708, 0.0103769, 0.0080708, 0.0103769, -0.0023061, 0.0023061)
7: (-0.0222632, -0.0131032, -0.0222632, -0.0131032, -0.0078697, 0.0078697)
8: (0.9600043, 0.9862487, 0.9600043, 0.9862487, -0.0252182, 0.0252182)
9: (0.0016947, 0.0094079, 0.0016947, 0.0094079, -0.0068188, 0.0068188)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.91 = 3.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0169093, upper bound: 0.0169093

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163998, upper bound: 0.0159995
time: 0.86 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0163998, upper bound: 0.0163998
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.86 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 8, lower bound: -0.0163998, upper bound: 0.0159995
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 8, lower bound: -0.0163998, upper bound: 0.0163998

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0005461, 0.0007755, -0.0005025, 0.0007669, -0.0011627, 0.0011159
1: -0.0012362, 0.0025102, -0.0010321, 0.0024970, -0.0036808, 0.0033339
2: 0.0125808, 0.0181913, 0.0126005, 0.0178857, -0.0048253, 0.0053320
3: -0.0011667, 0.0030522, -0.0011519, 0.0028224, -0.0035627, 0.0039337
4: -0.0054558, -0.0015643, -0.0054421, -0.0017763, -0.0036796, 0.0038778
5: 0.0067736, 0.0109848, 0.0067884, 0.0107554, -0.0035504, 0.0039194
6: 0.0080715, 0.0103767, 0.0080914, 0.0103712, -0.0022997, 0.0022853
7: -0.0222462, -0.0131042, -0.0217482, -0.0131363, -0.0078178, 0.0071208
8: 0.9600530, 0.9862459, 0.9614796, 0.9861538, -0.0250722, 0.0226914
9: 0.0016955, 0.0093936, 0.0017225, 0.0089743, -0.0061658, 0.0067757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0159995
time: 0.81 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0159995
time: 0.95 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0005439, 0.0007752, -0.0005185, 0.0008830, -0.0013001, 0.0011307
1: -0.0012261, 0.0025096, -0.0011071, 0.0026749, -0.0039010, 0.0035134
2: 0.0125816, 0.0181762, 0.0123340, 0.0179980, -0.0050486, 0.0057703
3: -0.0011661, 0.0030408, -0.0013522, 0.0029068, -0.0037089, 0.0042639
4: -0.0054552, -0.0015748, -0.0056269, -0.0016984, -0.0037569, 0.0040522
5: 0.0067742, 0.0109735, 0.0065884, 0.0108397, -0.0036944, 0.0042492
6: 0.0080723, 0.0103765, 0.0078216, 0.0104466, -0.0023743, 0.0025549
7: -0.0222216, -0.0131056, -0.0219312, -0.0127022, -0.0085354, 0.0073122
8: 0.9601233, 0.9862419, 0.9609553, 0.9873976, -0.0271188, 0.0237858
9: 0.0016966, 0.0093729, 0.0013570, 0.0091284, -0.0063555, 0.0073790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0163998
time: 0.95 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0163998
time: 1.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.24 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0159995
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0159995
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0163998
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 8, lower bound: -0.0159995, upper bound: 0.0163998

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005025, 0.0007669, -0.0005025, 0.0007669, -0.0011073, 0.0011073
1: -0.0010321, 0.0024970, -0.0010321, 0.0024970, -0.0033217, 0.0033217
2: 0.0126005, 0.0178857, 0.0126005, 0.0178857, -0.0048071, 0.0048071
3: -0.0011519, 0.0028224, -0.0011519, 0.0028224, -0.0035490, 0.0035490
4: -0.0054421, -0.0017763, -0.0054421, -0.0017763, -0.0036659, 0.0036659
5: 0.0067884, 0.0107554, 0.0067884, 0.0107554, -0.0035367, 0.0035367
6: 0.0080914, 0.0103712, 0.0080914, 0.0103712, -0.0022797, 0.0022797
7: -0.0217482, -0.0131363, -0.0217482, -0.0131363, -0.0070911, 0.0070911
8: 0.9614796, 0.9861538, 0.9614796, 0.9861538, -0.0226062, 0.0226062
9: 0.0017225, 0.0089743, 0.0017225, 0.0089743, -0.0061408, 0.0061408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156428, upper bound: 0.0155817
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156887, upper bound: 0.0156631
time: 0.77 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005185, 0.0008830, -0.0005025, 0.0007669, -0.0011381, 0.0012406
1: -0.0011071, 0.0026749, -0.0010321, 0.0024970, -0.0034884, 0.0035886
2: 0.0123340, 0.0179980, 0.0126005, 0.0178857, -0.0052069, 0.0050485
3: -0.0013522, 0.0029068, -0.0011519, 0.0028224, -0.0038496, 0.0037267
4: -0.0056269, -0.0016984, -0.0054421, -0.0017763, -0.0038507, 0.0037438
5: 0.0065884, 0.0108397, 0.0067884, 0.0107554, -0.0038367, 0.0037139
6: 0.0078216, 0.0104466, 0.0080914, 0.0103712, -0.0025495, 0.0023552
7: -0.0219312, -0.0127022, -0.0217482, -0.0131363, -0.0074479, 0.0077425
8: 0.9609553, 0.9873976, 0.9614796, 0.9861538, -0.0237443, 0.0244726
9: 0.0013570, 0.0091284, 0.0017225, 0.0089743, -0.0066893, 0.0064584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156428, upper bound: 0.0155817
time: 1.10 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156887, upper bound: 0.0156631
time: 0.93 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005025, 0.0007669, -0.0005185, 0.0008830, -0.0012406, 0.0011381
1: -0.0010321, 0.0024970, -0.0011071, 0.0026749, -0.0035886, 0.0034884
2: 0.0126005, 0.0178857, 0.0123340, 0.0179980, -0.0050485, 0.0052069
3: -0.0011519, 0.0028224, -0.0013522, 0.0029068, -0.0037267, 0.0038496
4: -0.0054421, -0.0017763, -0.0056269, -0.0016984, -0.0037438, 0.0038507
5: 0.0067884, 0.0107554, 0.0065884, 0.0108397, -0.0037139, 0.0038367
6: 0.0080914, 0.0103712, 0.0078216, 0.0104466, -0.0023552, 0.0025495
7: -0.0217482, -0.0131363, -0.0219312, -0.0127022, -0.0077425, 0.0074479
8: 0.9614796, 0.9861538, 0.9609553, 0.9873976, -0.0244726, 0.0237443
9: 0.0017225, 0.0089743, 0.0013570, 0.0091284, -0.0064584, 0.0066893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0155817, upper bound: 0.0160155
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156631, upper bound: 0.0160690
time: 0.77 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005185, 0.0008830, -0.0005185, 0.0008830, -0.0012203, 0.0012203
1: -0.0011071, 0.0026749, -0.0011071, 0.0026749, -0.0035830, 0.0035830
2: 0.0123340, 0.0179980, 0.0123340, 0.0179980, -0.0051528, 0.0051528
3: -0.0013522, 0.0029068, -0.0013522, 0.0029068, -0.0037872, 0.0037872
4: -0.0056269, -0.0016984, -0.0056269, -0.0016984, -0.0039286, 0.0039286
5: 0.0065884, 0.0108397, 0.0065884, 0.0108397, -0.0037726, 0.0037726
6: 0.0078216, 0.0104466, 0.0078216, 0.0104466, -0.0026250, 0.0026250
7: -0.0219312, -0.0127022, -0.0219312, -0.0127022, -0.0074820, 0.0074820
8: 0.9609553, 0.9873976, 0.9609553, 0.9873976, -0.0242721, 0.0242721
9: 0.0013570, 0.0091284, 0.0013570, 0.0091284, -0.0064984, 0.0064984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156198, upper bound: 0.0155830
time: 1.00 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156631, upper bound: 0.0156668
time: 1.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.36 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0156428, upper bound: 0.0155817
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0156887, upper bound: 0.0156631
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0156428, upper bound: 0.0155817
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0156887, upper bound: 0.0156631
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0155817, upper bound: 0.0160155
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0156631, upper bound: 0.0160690
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0156198, upper bound: 0.0155830
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 8, lower bound: -0.0156631, upper bound: 0.0156668

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0005123, 0.0006789, -0.0005018, 0.0007531, -0.0011030, 0.0010200
1: -0.0010782, 0.0023621, -0.0010287, 0.0024758, -0.0032467, 0.0031580
2: 0.0128025, 0.0179547, 0.0126322, 0.0178805, -0.0045666, 0.0047270
3: -0.0010000, 0.0028743, -0.0011280, 0.0028185, -0.0033701, 0.0035024
4: -0.0053020, -0.0017284, -0.0054201, -0.0017798, -0.0035222, 0.0036917
5: 0.0069400, 0.0108072, 0.0068122, 0.0107515, -0.0033582, 0.0034914
6: 0.0082960, 0.0103139, 0.0081236, 0.0103622, -0.0020661, 0.0021903
7: -0.0218607, -0.0134655, -0.0217398, -0.0131881, -0.0071133, 0.0067185
8: 0.9611574, 0.9852107, 0.9615037, 0.9860055, -0.0221980, 0.0214791
9: 0.0019997, 0.0090690, 0.0017661, 0.0089672, -0.0058231, 0.0061243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152815, upper bound: 0.0153686
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154027, upper bound: 0.0153686
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0005004, 0.0007120, -0.0005025, 0.0007669, -0.0011035, 0.0010630
1: -0.0010224, 0.0024128, -0.0010321, 0.0024970, -0.0032389, 0.0032898
2: 0.0127265, 0.0178711, 0.0126005, 0.0178857, -0.0047544, 0.0047060
3: -0.0010571, 0.0028114, -0.0011519, 0.0028224, -0.0035052, 0.0034828
4: -0.0053547, -0.0017864, -0.0054421, -0.0017763, -0.0035785, 0.0036558
5: 0.0068830, 0.0107445, 0.0067884, 0.0107554, -0.0034926, 0.0034713
6: 0.0082190, 0.0103355, 0.0080914, 0.0103712, -0.0021521, 0.0022440
7: -0.0217245, -0.0133417, -0.0217482, -0.0131363, -0.0070285, 0.0069904
8: 0.9615476, 0.9855654, 0.9614796, 0.9861538, -0.0221105, 0.0223681
9: 0.0018955, 0.0089543, 0.0017225, 0.0089743, -0.0060550, 0.0060647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153095, upper bound: 0.0154415
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154415, upper bound: 0.0154415
time: 0.93 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0005289, 0.0007937, -0.0005018, 0.0007531, -0.0011333, 0.0011513
1: -0.0011555, 0.0025381, -0.0010287, 0.0024758, -0.0034224, 0.0034282
2: 0.0125389, 0.0180705, 0.0126322, 0.0178805, -0.0049712, 0.0049819
3: -0.0011981, 0.0029614, -0.0011280, 0.0028185, -0.0036743, 0.0036910
4: -0.0054848, -0.0016481, -0.0054201, -0.0017798, -0.0037050, 0.0037720
5: 0.0067422, 0.0108941, 0.0068122, 0.0107515, -0.0036619, 0.0036794
6: 0.0080291, 0.0103886, 0.0081236, 0.0103622, -0.0023330, 0.0022650
7: -0.0220494, -0.0130361, -0.0217398, -0.0131881, -0.0074718, 0.0073777
8: 0.9606167, 0.9864410, 0.9615037, 0.9860055, -0.0233976, 0.0233678
9: 0.0016381, 0.0092279, 0.0017661, 0.0089672, -0.0063782, 0.0064460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156563, upper bound: 0.0153444
time: 0.82 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157608, upper bound: 0.0153444
time: 1.09 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0005163, 0.0008282, -0.0005025, 0.0007669, -0.0011343, 0.0011954
1: -0.0010967, 0.0025909, -0.0010321, 0.0024970, -0.0034101, 0.0035597
2: 0.0124599, 0.0179825, 0.0126005, 0.0178857, -0.0051585, 0.0049513
3: -0.0012576, 0.0028952, -0.0011519, 0.0028224, -0.0038091, 0.0036635
4: -0.0055396, -0.0017091, -0.0054421, -0.0017763, -0.0037634, 0.0037330
5: 0.0066829, 0.0108280, 0.0067884, 0.0107554, -0.0037960, 0.0036516
6: 0.0079490, 0.0104110, 0.0080914, 0.0103712, -0.0024221, 0.0023196
7: -0.0219059, -0.0129073, -0.0217482, -0.0131363, -0.0073875, 0.0076489
8: 0.9610278, 0.9868101, 0.9614796, 0.9861538, -0.0232664, 0.0242548
9: 0.0015297, 0.0091071, 0.0017225, 0.0089743, -0.0066095, 0.0063842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158146, upper bound: 0.0152815
time: 0.98 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0158146, upper bound: 0.0154204
time: 1.03 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005018, 0.0007531, -0.0005289, 0.0007937, -0.0011513, 0.0011333
1: -0.0010287, 0.0024758, -0.0011555, 0.0025381, -0.0034282, 0.0034224
2: 0.0126322, 0.0178805, 0.0125389, 0.0180705, -0.0049819, 0.0049712
3: -0.0011280, 0.0028185, -0.0011981, 0.0029614, -0.0036910, 0.0036743
4: -0.0054201, -0.0017798, -0.0054848, -0.0016481, -0.0037720, 0.0037050
5: 0.0068122, 0.0107515, 0.0067422, 0.0108941, -0.0036794, 0.0036619
6: 0.0081236, 0.0103622, 0.0080291, 0.0103886, -0.0022650, 0.0023330
7: -0.0217398, -0.0131881, -0.0220494, -0.0130361, -0.0073777, 0.0074718
8: 0.9615037, 0.9860055, 0.9606167, 0.9864410, -0.0233678, 0.0233976
9: 0.0017661, 0.0089672, 0.0016381, 0.0092279, -0.0064460, 0.0063782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153444, upper bound: 0.0156563
time: 0.88 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153444, upper bound: 0.0157608
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005025, 0.0007669, -0.0005163, 0.0008282, -0.0011954, 0.0011343
1: -0.0010321, 0.0024970, -0.0010967, 0.0025909, -0.0035597, 0.0034101
2: 0.0126005, 0.0178857, 0.0124599, 0.0179825, -0.0049513, 0.0051585
3: -0.0011519, 0.0028224, -0.0012576, 0.0028952, -0.0036635, 0.0038091
4: -0.0054421, -0.0017763, -0.0055396, -0.0017091, -0.0037330, 0.0037634
5: 0.0067884, 0.0107554, 0.0066829, 0.0108280, -0.0036516, 0.0037960
6: 0.0080914, 0.0103712, 0.0079490, 0.0104110, -0.0023196, 0.0024221
7: -0.0217482, -0.0131363, -0.0219059, -0.0129073, -0.0076489, 0.0073875
8: 0.9614796, 0.9861538, 0.9610278, 0.9868101, -0.0242548, 0.0232664
9: 0.0017225, 0.0089743, 0.0015297, 0.0091071, -0.0063842, 0.0066095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152815, upper bound: 0.0158146
time: 0.99 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0154204, upper bound: 0.0158146
time: 0.86 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0005289, 0.0007937, -0.0005177, 0.0008691, -0.0012158, 0.0011306
1: -0.0011555, 0.0025381, -0.0011035, 0.0026536, -0.0034955, 0.0034200
2: 0.0125389, 0.0180705, 0.0123660, 0.0179926, -0.0049131, 0.0050572
3: -0.0011981, 0.0029614, -0.0013282, 0.0029028, -0.0036091, 0.0037322
4: -0.0054848, -0.0016481, -0.0056048, -0.0017021, -0.0037827, 0.0039567
5: 0.0067422, 0.0108941, 0.0066124, 0.0108356, -0.0035951, 0.0037192
6: 0.0080291, 0.0103886, 0.0078540, 0.0104376, -0.0024084, 0.0025346
7: -0.0220494, -0.0130361, -0.0219224, -0.0127542, -0.0075018, 0.0071126
8: 0.9606167, 0.9864410, 0.9609807, 0.9872485, -0.0237829, 0.0231474
9: 0.0016381, 0.0092279, 0.0014008, 0.0091210, -0.0061831, 0.0064761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0156563, upper bound: 0.0153444
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157608, upper bound: 0.0153444
time: 1.04 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0005163, 0.0008282, -0.0005185, 0.0008830, -0.0012166, 0.0011760
1: -0.0010967, 0.0025909, -0.0011071, 0.0026749, -0.0034977, 0.0035527
2: 0.0124599, 0.0179825, 0.0123340, 0.0179980, -0.0051052, 0.0050457
3: -0.0012576, 0.0028952, -0.0013522, 0.0029068, -0.0037501, 0.0037177
4: -0.0055396, -0.0017091, -0.0056269, -0.0016984, -0.0038413, 0.0039178
5: 0.0066829, 0.0108280, 0.0065884, 0.0108397, -0.0037353, 0.0037042
6: 0.0079490, 0.0104110, 0.0078216, 0.0104466, -0.0024976, 0.0025893
7: -0.0219059, -0.0129073, -0.0219312, -0.0127022, -0.0074195, 0.0073952
8: 0.9610278, 0.9868101, 0.9609553, 0.9873976, -0.0237464, 0.0240526
9: 0.0015297, 0.0091071, 0.0013570, 0.0091284, -0.0064239, 0.0064205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159678, upper bound: 0.0156226
time: 0.89 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0159678, upper bound: 0.0156668
time: 1.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.16 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0152815, upper bound: 0.0153686
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0154027, upper bound: 0.0153686
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0153095, upper bound: 0.0154415
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0154415, upper bound: 0.0154415
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0156563, upper bound: 0.0153444
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0157608, upper bound: 0.0153444
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0158146, upper bound: 0.0152815
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0158146, upper bound: 0.0154204
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0153444, upper bound: 0.0156563
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0153444, upper bound: 0.0157608
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0152815, upper bound: 0.0158146
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0154204, upper bound: 0.0158146
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0156563, upper bound: 0.0153444
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0157608, upper bound: 0.0153444
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0159678, upper bound: 0.0156226
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 8, lower bound: -0.0159678, upper bound: 0.0156668

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005117, 0.0006618, -0.0005023, 0.0006410, -0.0009891, 0.0009997
1: -0.0010750, 0.0023358, -0.0010310, 0.0023040, -0.0030496, 0.0030390
2: 0.0128418, 0.0179499, 0.0128895, 0.0178841, -0.0044223, 0.0044386
3: -0.0009704, 0.0028707, -0.0009345, 0.0028212, -0.0032733, 0.0032875
4: -0.0052747, -0.0017317, -0.0052416, -0.0017774, -0.0034973, 0.0035099
5: 0.0069695, 0.0108036, 0.0070054, 0.0107542, -0.0032626, 0.0032769
6: 0.0083358, 0.0103028, 0.0083842, 0.0102893, -0.0019534, 0.0019186
7: -0.0218529, -0.0135296, -0.0217456, -0.0136074, -0.0066617, 0.0065982
8: 0.9611799, 0.9850270, 0.9614872, 0.9848043, -0.0208431, 0.0207709
9: 0.0020537, 0.0090624, 0.0021192, 0.0089721, -0.0056995, 0.0057405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144456, upper bound: 0.0145313
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005123, 0.0006789, -0.0004998, 0.0007101, -0.0010446, 0.0010180
1: -0.0010782, 0.0023621, -0.0010193, 0.0024099, -0.0031294, 0.0031101
2: 0.0128025, 0.0179547, 0.0127309, 0.0178665, -0.0045160, 0.0045387
3: -0.0010000, 0.0028743, -0.0010538, 0.0028080, -0.0033380, 0.0033529
4: -0.0053020, -0.0017284, -0.0053517, -0.0017896, -0.0035125, 0.0036233
5: 0.0069400, 0.0108072, 0.0068863, 0.0107410, -0.0033267, 0.0033414
6: 0.0082960, 0.0103139, 0.0082235, 0.0103342, -0.0020382, 0.0020904
7: -0.0218607, -0.0134655, -0.0217170, -0.0133489, -0.0067567, 0.0066882
8: 0.9611574, 0.9852107, 0.9615691, 0.9855448, -0.0213367, 0.0212216
9: 0.0019997, 0.0090690, 0.0019015, 0.0089480, -0.0057867, 0.0058245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146737, upper bound: 0.0146894
time: 0.81 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145749, upper bound: 0.0145323
time: 0.78 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004997, 0.0006953, -0.0005030, 0.0006556, -0.0009902, 0.0010374
1: -0.0010189, 0.0023872, -0.0010343, 0.0023263, -0.0030426, 0.0031555
2: 0.0127650, 0.0178660, 0.0128560, 0.0178890, -0.0045834, 0.0044189
3: -0.0010282, 0.0028076, -0.0009597, 0.0028248, -0.0033882, 0.0032688
4: -0.0053280, -0.0017899, -0.0052649, -0.0017740, -0.0035540, 0.0034749
5: 0.0069119, 0.0107406, 0.0069802, 0.0107579, -0.0033766, 0.0032578
6: 0.0082580, 0.0103246, 0.0083502, 0.0102988, -0.0020408, 0.0019743
7: -0.0217161, -0.0134044, -0.0217536, -0.0135528, -0.0065794, 0.0068045
8: 0.9615716, 0.9853858, 0.9614643, 0.9849607, -0.0207602, 0.0215379
9: 0.0019482, 0.0089473, 0.0020732, 0.0089788, -0.0058759, 0.0056825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147058, upper bound: 0.0148799
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146501, upper bound: 0.0147744
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005004, 0.0007120, -0.0005005, 0.0007240, -0.0010487, 0.0010612
1: -0.0010224, 0.0024128, -0.0010227, 0.0024311, -0.0031394, 0.0032414
2: 0.0127265, 0.0178711, 0.0126991, 0.0178715, -0.0047020, 0.0045406
3: -0.0010571, 0.0028114, -0.0010777, 0.0028118, -0.0034731, 0.0033529
4: -0.0053547, -0.0017864, -0.0053737, -0.0017861, -0.0035686, 0.0035874
5: 0.0068830, 0.0107445, 0.0068624, 0.0107448, -0.0034612, 0.0033413
6: 0.0082190, 0.0103355, 0.0081913, 0.0103432, -0.0021242, 0.0021442
7: -0.0217245, -0.0133417, -0.0217252, -0.0132970, -0.0067225, 0.0069618
8: 0.9615476, 0.9855654, 0.9615457, 0.9856933, -0.0213538, 0.0221032
9: 0.0018955, 0.0089543, 0.0018579, 0.0089549, -0.0060206, 0.0058136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148416, upper bound: 0.0148799
time: 0.82 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147744, upper bound: 0.0147744
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005282, 0.0007765, -0.0005023, 0.0006410, -0.0010193, 0.0011318
1: -0.0011523, 0.0025117, -0.0010310, 0.0023040, -0.0032258, 0.0033132
2: 0.0125785, 0.0180657, 0.0128895, 0.0178841, -0.0048330, 0.0046932
3: -0.0011684, 0.0029577, -0.0009345, 0.0028212, -0.0035821, 0.0034768
4: -0.0054574, -0.0016514, -0.0052416, -0.0017774, -0.0036800, 0.0035902
5: 0.0067719, 0.0108905, 0.0070054, 0.0107542, -0.0035708, 0.0034657
6: 0.0080691, 0.0103774, 0.0083842, 0.0102893, -0.0022201, 0.0019932
7: -0.0220415, -0.0131005, -0.0217456, -0.0136074, -0.0070203, 0.0072673
8: 0.9606394, 0.9862564, 0.9614872, 0.9848043, -0.0220436, 0.0226881
9: 0.0016924, 0.0092213, 0.0021192, 0.0089721, -0.0062629, 0.0060619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
time: 0.94 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005289, 0.0007937, -0.0004998, 0.0007101, -0.0010786, 0.0011493
1: -0.0011555, 0.0025381, -0.0010193, 0.0024099, -0.0033164, 0.0033803
2: 0.0125389, 0.0180705, 0.0127309, 0.0178665, -0.0049205, 0.0048158
3: -0.0011981, 0.0029614, -0.0010538, 0.0028080, -0.0036422, 0.0035612
4: -0.0054848, -0.0016481, -0.0053517, -0.0017896, -0.0036952, 0.0037036
5: 0.0067422, 0.0108941, 0.0068863, 0.0107410, -0.0036303, 0.0035494
6: 0.0080291, 0.0103886, 0.0082235, 0.0103342, -0.0023051, 0.0021651
7: -0.0220494, -0.0130361, -0.0217170, -0.0133489, -0.0071677, 0.0073474
8: 0.9606167, 0.9864410, 0.9615691, 0.9855448, -0.0226290, 0.0231103
9: 0.0016381, 0.0092279, 0.0019015, 0.0089480, -0.0063417, 0.0061951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149604, upper bound: 0.0146675
time: 0.85 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
time: 0.81 seconds

## BFS IS instance: IS_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0005179, 0.0007172, -0.0005017, 0.0007498, -0.0011173, 0.0010843
1: -0.0011043, 0.0024207, -0.0010286, 0.0024707, -0.0033129, 0.0033767
2: 0.0127147, 0.0179937, 0.0126398, 0.0178804, -0.0048899, 0.0048429
3: -0.0010660, 0.0029036, -0.0011223, 0.0028184, -0.0036105, 0.0035937
4: -0.0053629, -0.0017013, -0.0054149, -0.0017799, -0.0035830, 0.0037136
5: 0.0068741, 0.0108365, 0.0068179, 0.0107514, -0.0035980, 0.0035826
6: 0.0082071, 0.0103388, 0.0081312, 0.0103600, -0.0021529, 0.0022076
7: -0.0219243, -0.0133224, -0.0217396, -0.0132004, -0.0073121, 0.0072347
8: 0.9609751, 0.9856206, 0.9615042, 0.9859702, -0.0227231, 0.0229953
9: 0.0018792, 0.0091226, 0.0017765, 0.0089671, -0.0062567, 0.0062974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152524
time: 0.97 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152815
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0005144, 0.0007847, -0.0005025, 0.0007669, -0.0011325, 0.0011358
1: -0.0010879, 0.0025242, -0.0010321, 0.0024970, -0.0033644, 0.0034500
2: 0.0125597, 0.0179693, 0.0126005, 0.0178857, -0.0049808, 0.0049030
3: -0.0011825, 0.0028852, -0.0011519, 0.0028224, -0.0036684, 0.0036351
4: -0.0054704, -0.0017183, -0.0054421, -0.0017763, -0.0036941, 0.0037239
5: 0.0067578, 0.0108182, 0.0067884, 0.0107554, -0.0036550, 0.0036236
6: 0.0080502, 0.0103827, 0.0080914, 0.0103712, -0.0023210, 0.0022913
7: -0.0218844, -0.0130699, -0.0217482, -0.0131363, -0.0073593, 0.0073002
8: 0.9610892, 0.9863440, 0.9614796, 0.9861538, -0.0230191, 0.0234379
9: 0.0016666, 0.0090890, 0.0017225, 0.0089743, -0.0063161, 0.0063507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0153806
time: 0.97 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0154204
time: 1.02 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005023, 0.0006410, -0.0005282, 0.0007765, -0.0011318, 0.0010193
1: -0.0010310, 0.0023040, -0.0011523, 0.0025117, -0.0033132, 0.0032258
2: 0.0128895, 0.0178841, 0.0125785, 0.0180657, -0.0046932, 0.0048330
3: -0.0009345, 0.0028212, -0.0011684, 0.0029577, -0.0034768, 0.0035821
4: -0.0052416, -0.0017774, -0.0054574, -0.0016514, -0.0035902, 0.0036800
5: 0.0070054, 0.0107542, 0.0067719, 0.0108905, -0.0034657, 0.0035708
6: 0.0083842, 0.0102893, 0.0080691, 0.0103774, -0.0019932, 0.0022201
7: -0.0217456, -0.0136074, -0.0220415, -0.0131005, -0.0072673, 0.0070203
8: 0.9614872, 0.9848043, 0.9606394, 0.9862564, -0.0226881, 0.0220436
9: 0.0021192, 0.0089721, 0.0016924, 0.0092213, -0.0060619, 0.0062629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146675, upper bound: 0.0148443
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145069, upper bound: 0.0146978
time: 0.84 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004998, 0.0007101, -0.0005289, 0.0007937, -0.0011493, 0.0010786
1: -0.0010193, 0.0024099, -0.0011555, 0.0025381, -0.0033803, 0.0033164
2: 0.0127309, 0.0178665, 0.0125389, 0.0180705, -0.0048158, 0.0049205
3: -0.0010538, 0.0028080, -0.0011981, 0.0029614, -0.0035612, 0.0036422
4: -0.0053517, -0.0017896, -0.0054848, -0.0016481, -0.0037036, 0.0036952
5: 0.0068863, 0.0107410, 0.0067422, 0.0108941, -0.0035494, 0.0036303
6: 0.0082235, 0.0103342, 0.0080291, 0.0103886, -0.0021651, 0.0023051
7: -0.0217170, -0.0133489, -0.0220494, -0.0130361, -0.0073474, 0.0071677
8: 0.9615691, 0.9855448, 0.9606167, 0.9864410, -0.0231103, 0.0226290
9: 0.0019015, 0.0089480, 0.0016381, 0.0092279, -0.0061951, 0.0063417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146675, upper bound: 0.0149604
time: 0.81 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145070, upper bound: 0.0148265
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005017, 0.0007498, -0.0005179, 0.0007172, -0.0010843, 0.0011173
1: -0.0010286, 0.0024707, -0.0011043, 0.0024207, -0.0033767, 0.0033129
2: 0.0126398, 0.0178804, 0.0127147, 0.0179937, -0.0048429, 0.0048899
3: -0.0011223, 0.0028184, -0.0010660, 0.0029036, -0.0035937, 0.0036105
4: -0.0054149, -0.0017799, -0.0053629, -0.0017013, -0.0037136, 0.0035830
5: 0.0068179, 0.0107514, 0.0068741, 0.0108365, -0.0035826, 0.0035980
6: 0.0081312, 0.0103600, 0.0082071, 0.0103388, -0.0022076, 0.0021529
7: -0.0217396, -0.0132004, -0.0219243, -0.0133224, -0.0072347, 0.0073121
8: 0.9615042, 0.9859702, 0.9609751, 0.9856206, -0.0229953, 0.0227231
9: 0.0017765, 0.0089671, 0.0018792, 0.0091226, -0.0062974, 0.0062567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151760, upper bound: 0.0157250
time: 1.03 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151760, upper bound: 0.0158146
time: 1.05 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005025, 0.0007669, -0.0005144, 0.0007847, -0.0011358, 0.0011325
1: -0.0010321, 0.0024970, -0.0010879, 0.0025242, -0.0034500, 0.0033644
2: 0.0126005, 0.0178857, 0.0125597, 0.0179693, -0.0049030, 0.0049808
3: -0.0011519, 0.0028224, -0.0011825, 0.0028852, -0.0036351, 0.0036684
4: -0.0054421, -0.0017763, -0.0054704, -0.0017183, -0.0037239, 0.0036941
5: 0.0067884, 0.0107554, 0.0067578, 0.0108182, -0.0036236, 0.0036550
6: 0.0080914, 0.0103712, 0.0080502, 0.0103827, -0.0022913, 0.0023210
7: -0.0217482, -0.0131363, -0.0218844, -0.0130699, -0.0073002, 0.0073593
8: 0.9614796, 0.9861538, 0.9610892, 0.9863440, -0.0234379, 0.0230191
9: 0.0017225, 0.0089743, 0.0016666, 0.0090890, -0.0063507, 0.0063161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153413, upper bound: 0.0157250
time: 1.16 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0153413, upper bound: 0.0158146
time: 0.97 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005282, 0.0007765, -0.0005193, 0.0007555, -0.0011026, 0.0011115
1: -0.0011523, 0.0025117, -0.0011109, 0.0024794, -0.0033061, 0.0032920
2: 0.0125785, 0.0180657, 0.0126268, 0.0180036, -0.0047591, 0.0047798
3: -0.0011684, 0.0029577, -0.0011320, 0.0029111, -0.0035079, 0.0035269
4: -0.0054574, -0.0016514, -0.0054238, -0.0016945, -0.0037629, 0.0037724
5: 0.0067719, 0.0108905, 0.0068082, 0.0108439, -0.0034951, 0.0035145
6: 0.0080691, 0.0103774, 0.0081181, 0.0103637, -0.0022945, 0.0022592
7: -0.0220415, -0.0131005, -0.0219404, -0.0131793, -0.0070732, 0.0069997
8: 0.9606394, 0.9862564, 0.9609290, 0.9860306, -0.0224818, 0.0223887
9: 0.0016924, 0.0092213, 0.0017587, 0.0091361, -0.0060591, 0.0061113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146677
time: 0.97 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
time: 0.98 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005289, 0.0007937, -0.0005158, 0.0008252, -0.0011579, 0.0011287
1: -0.0011555, 0.0025381, -0.0010945, 0.0025863, -0.0033925, 0.0033669
2: 0.0125389, 0.0180705, 0.0124667, 0.0179791, -0.0048541, 0.0048866
3: -0.0011981, 0.0029614, -0.0012525, 0.0028927, -0.0035750, 0.0035960
4: -0.0054848, -0.0016481, -0.0055349, -0.0017114, -0.0037734, 0.0038869
5: 0.0067422, 0.0108941, 0.0066880, 0.0108256, -0.0035616, 0.0035825
6: 0.0080291, 0.0103886, 0.0079559, 0.0104090, -0.0023799, 0.0024326
7: -0.0220494, -0.0130361, -0.0219005, -0.0129183, -0.0071620, 0.0070827
8: 0.9606167, 0.9864410, 0.9610433, 0.9867784, -0.0230036, 0.0228460
9: 0.0016381, 0.0092279, 0.0015390, 0.0091026, -0.0061463, 0.0061893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149604, upper bound: 0.0146677
time: 0.81 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
time: 0.88 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005163, 0.0008282, -0.0005289, 0.0007937, -0.0011279, 0.0011763
1: -0.0010967, 0.0025909, -0.0011555, 0.0025381, -0.0033525, 0.0034327
2: 0.0124599, 0.0179825, 0.0125389, 0.0180705, -0.0049631, 0.0048283
3: -0.0012576, 0.0028952, -0.0011981, 0.0029614, -0.0036614, 0.0035541
4: -0.0055396, -0.0017091, -0.0054848, -0.0016481, -0.0038916, 0.0037757
5: 0.0066829, 0.0108280, 0.0067422, 0.0108941, -0.0036486, 0.0035410
6: 0.0079490, 0.0104110, 0.0080291, 0.0103886, -0.0024395, 0.0023818
7: -0.0219059, -0.0129073, -0.0220494, -0.0130361, -0.0070652, 0.0073485
8: 0.9610278, 0.9868101, 0.9606167, 0.9864410, -0.0227311, 0.0233436
9: 0.0015297, 0.0091071, 0.0016381, 0.0092279, -0.0063470, 0.0061221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152526
time: 1.11 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0153806
time: 1.08 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005163, 0.0008282, -0.0005163, 0.0008282, -0.0011728, 0.0011728
1: -0.0010967, 0.0025909, -0.0010967, 0.0025909, -0.0034904, 0.0034904
2: 0.0124599, 0.0179825, 0.0124599, 0.0179825, -0.0050291, 0.0050291
3: -0.0012576, 0.0028952, -0.0012576, 0.0028952, -0.0037001, 0.0037001
4: -0.0055396, -0.0017091, -0.0055396, -0.0017091, -0.0038305, 0.0038305
5: 0.0066829, 0.0108280, 0.0066829, 0.0108280, -0.0036862, 0.0036862
6: 0.0079490, 0.0104110, 0.0079490, 0.0104110, -0.0024619, 0.0024619
7: -0.0219059, -0.0129073, -0.0219059, -0.0129073, -0.0073459, 0.0073459
8: 0.9610278, 0.9868101, 0.9610278, 0.9868101, -0.0236752, 0.0236752
9: 0.0015297, 0.0091071, 0.0015297, 0.0091071, -0.0063656, 0.0063656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152824
time: 1.18 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0154213
time: 1.12 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.53 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0144456, upper bound: 0.0145313
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0146737, upper bound: 0.0146894
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0145749, upper bound: 0.0145323
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0147058, upper bound: 0.0148799
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0146501, upper bound: 0.0147744
IS_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0148416, upper bound: 0.0148799
IS_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0147744, upper bound: 0.0147744
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0149604, upper bound: 0.0146675
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
IS_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152524
IS_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152815
IS_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0153806
IS_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0154204
IS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0146675, upper bound: 0.0148443
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0145069, upper bound: 0.0146978
IS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0146675, upper bound: 0.0149604
IS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0145070, upper bound: 0.0148265
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0151760, upper bound: 0.0157250
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0151760, upper bound: 0.0158146
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0153413, upper bound: 0.0157250
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0153413, upper bound: 0.0158146
IS_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146677
IS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
IS_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0149604, upper bound: 0.0146677
IS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152526
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0153806
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0152824
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 8, lower bound: -0.0157250, upper bound: 0.0154213

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005117, 0.0006618, -0.0004830, 0.0006363, -0.0009843, 0.0009792
1: -0.0010750, 0.0023358, -0.0009410, 0.0022967, -0.0030423, 0.0029474
2: 0.0128418, 0.0179499, 0.0129004, 0.0177492, -0.0042807, 0.0044277
3: -0.0009704, 0.0028707, -0.0009264, 0.0027198, -0.0031636, 0.0032792
4: -0.0052747, -0.0017317, -0.0052341, -0.0018709, -0.0034038, 0.0035024
5: 0.0069695, 0.0108036, 0.0070135, 0.0106530, -0.0031528, 0.0032687
6: 0.0083358, 0.0103028, 0.0083951, 0.0102862, -0.0019504, 0.0019077
7: -0.0218529, -0.0135296, -0.0215259, -0.0136250, -0.0066438, 0.0063623
8: 0.9611799, 0.9850270, 0.9621167, 0.9847537, -0.0207920, 0.0201148
9: 0.0020537, 0.0090624, 0.0021340, 0.0087871, -0.0054961, 0.0057254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144456, upper bound: 0.0145313
time: 0.82 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144456, upper bound: 0.0145313
time: 0.82 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005023, 0.0006591, -0.0004528, 0.0007713, -0.0011138, 0.0009691
1: -0.0010312, 0.0023318, -0.0007994, 0.0025036, -0.0033646, 0.0030409
2: 0.0128479, 0.0178844, 0.0125905, 0.0175372, -0.0043564, 0.0048915
3: -0.0009658, 0.0028214, -0.0011593, 0.0025603, -0.0031974, 0.0036139
4: -0.0052705, -0.0017772, -0.0054490, -0.0020180, -0.0032525, 0.0036718
5: 0.0069741, 0.0107544, 0.0067809, 0.0104939, -0.0031844, 0.0036013
6: 0.0083420, 0.0103011, 0.0080814, 0.0103740, -0.0020319, 0.0022197
7: -0.0217461, -0.0135396, -0.0211804, -0.0131202, -0.0072129, 0.0062883
8: 0.9614856, 0.9849985, 0.9631065, 0.9862001, -0.0229849, 0.0205316
9: 0.0020621, 0.0089725, 0.0017089, 0.0084962, -0.0054632, 0.0062483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0144941
time: 1.00 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0145313
time: 0.96 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005123, 0.0006789, -0.0004804, 0.0007054, -0.0010398, 0.0009966
1: -0.0010782, 0.0023621, -0.0009287, 0.0024027, -0.0031221, 0.0030175
2: 0.0128025, 0.0179547, 0.0127416, 0.0177307, -0.0043701, 0.0045278
3: -0.0010000, 0.0028743, -0.0010457, 0.0027059, -0.0032275, 0.0033447
4: -0.0053020, -0.0017284, -0.0053442, -0.0018837, -0.0034183, 0.0036158
5: 0.0069400, 0.0108072, 0.0068943, 0.0106391, -0.0032162, 0.0033332
6: 0.0082960, 0.0103139, 0.0082344, 0.0103312, -0.0020352, 0.0020796
7: -0.0218607, -0.0134655, -0.0214958, -0.0133663, -0.0067389, 0.0064386
8: 0.9611574, 0.9852107, 0.9622030, 0.9854949, -0.0212858, 0.0205437
9: 0.0019997, 0.0090690, 0.0019162, 0.0087617, -0.0055820, 0.0058095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145233
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005030, 0.0006762, -0.0004513, 0.0008463, -0.0011745, 0.0009890
1: -0.0010345, 0.0023580, -0.0007925, 0.0026186, -0.0034598, 0.0031131
2: 0.0128086, 0.0178892, 0.0124184, 0.0175269, -0.0044538, 0.0050134
3: -0.0009954, 0.0028250, -0.0012888, 0.0025526, -0.0032659, 0.0036966
4: -0.0052978, -0.0017738, -0.0055684, -0.0020252, -0.0032726, 0.0037946
5: 0.0069446, 0.0107580, 0.0066517, 0.0104861, -0.0032527, 0.0036829
6: 0.0083022, 0.0103122, 0.0079070, 0.0104227, -0.0021205, 0.0024052
7: -0.0217540, -0.0134755, -0.0211635, -0.0128396, -0.0073246, 0.0063928
8: 0.9614632, 0.9851820, 0.9631548, 0.9870039, -0.0235748, 0.0209978
9: 0.0020081, 0.0089791, 0.0014727, 0.0084820, -0.0055657, 0.0063550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144981, upper bound: 0.0144981
time: 0.79 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144981, upper bound: 0.0145323
time: 0.97 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004997, 0.0006953, -0.0004837, 0.0006509, -0.0009854, 0.0010164
1: -0.0010189, 0.0023872, -0.0009441, 0.0023191, -0.0030352, 0.0030577
2: 0.0127650, 0.0178660, 0.0128669, 0.0177538, -0.0044281, 0.0044079
3: -0.0010282, 0.0028076, -0.0009515, 0.0027232, -0.0032686, 0.0032605
4: -0.0053280, -0.0017899, -0.0052573, -0.0018677, -0.0034603, 0.0034674
5: 0.0069119, 0.0107406, 0.0069884, 0.0106564, -0.0032572, 0.0032496
6: 0.0082580, 0.0103246, 0.0083613, 0.0102957, -0.0020377, 0.0019633
7: -0.0217161, -0.0134044, -0.0215333, -0.0135705, -0.0065616, 0.0065588
8: 0.9615716, 0.9853858, 0.9620953, 0.9849098, -0.0207091, 0.0208245
9: 0.0019482, 0.0089473, 0.0020881, 0.0087934, -0.0056630, 0.0056674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
time: 0.81 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004897, 0.0006926, -0.0004534, 0.0007853, -0.0011170, 0.0010051
1: -0.0009724, 0.0023831, -0.0008023, 0.0025251, -0.0033668, 0.0031451
2: 0.0127710, 0.0177962, 0.0125584, 0.0175416, -0.0045078, 0.0048774
3: -0.0010236, 0.0027551, -0.0011835, 0.0025636, -0.0033047, 0.0035975
4: -0.0053238, -0.0018383, -0.0054713, -0.0020150, -0.0033089, 0.0036330
5: 0.0069164, 0.0106882, 0.0067568, 0.0104971, -0.0032910, 0.0035844
6: 0.0082641, 0.0103228, 0.0080488, 0.0103831, -0.0021189, 0.0022740
7: -0.0216024, -0.0134142, -0.0211875, -0.0130678, -0.0071515, 0.0064718
8: 0.9618973, 0.9853576, 0.9630861, 0.9863502, -0.0229361, 0.0212476
9: 0.0019566, 0.0088515, 0.0016648, 0.0085021, -0.0056227, 0.0061975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0145747
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0147744
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005004, 0.0007120, -0.0004811, 0.0007193, -0.0010439, 0.0010385
1: -0.0010224, 0.0024128, -0.0009319, 0.0024240, -0.0031321, 0.0031427
2: 0.0127265, 0.0178711, 0.0127098, 0.0177355, -0.0045441, 0.0045296
3: -0.0010571, 0.0028114, -0.0010697, 0.0027095, -0.0033505, 0.0033446
4: -0.0053547, -0.0017864, -0.0053663, -0.0018804, -0.0034743, 0.0035799
5: 0.0068830, 0.0107445, 0.0068704, 0.0106427, -0.0033384, 0.0033330
6: 0.0082190, 0.0103355, 0.0082021, 0.0103402, -0.0021211, 0.0021333
7: -0.0217245, -0.0133417, -0.0215036, -0.0133145, -0.0067046, 0.0066956
8: 0.9615476, 0.9855654, 0.9621806, 0.9856433, -0.0213025, 0.0213737
9: 0.0018955, 0.0089543, 0.0018726, 0.0087683, -0.0057932, 0.0057986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146343, upper bound: 0.0147477
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146343, upper bound: 0.0148799
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004905, 0.0007094, -0.0004520, 0.0008593, -0.0011812, 0.0010304
1: -0.0009758, 0.0024088, -0.0007956, 0.0026385, -0.0034818, 0.0032044
2: 0.0127325, 0.0178013, 0.0123885, 0.0175314, -0.0046243, 0.0050325
3: -0.0010526, 0.0027589, -0.0013113, 0.0025560, -0.0033884, 0.0037062
4: -0.0053505, -0.0018348, -0.0055892, -0.0020220, -0.0033286, 0.0037544
5: 0.0068875, 0.0106921, 0.0066293, 0.0104895, -0.0033743, 0.0036922
6: 0.0082252, 0.0103337, 0.0078768, 0.0104312, -0.0022060, 0.0024570
7: -0.0216107, -0.0133515, -0.0211710, -0.0127909, -0.0073109, 0.0066439
8: 0.9618737, 0.9855372, 0.9631334, 0.9871435, -0.0236802, 0.0217995
9: 0.0019038, 0.0088585, 0.0014317, 0.0084883, -0.0057725, 0.0063492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145323, upper bound: 0.0145749
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145323, upper bound: 0.0147744
time: 0.95 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005282, 0.0007765, -0.0004830, 0.0006363, -0.0010145, 0.0011113
1: -0.0011523, 0.0025117, -0.0009410, 0.0022967, -0.0032185, 0.0032216
2: 0.0125785, 0.0180657, 0.0129004, 0.0177492, -0.0046914, 0.0046822
3: -0.0011684, 0.0029577, -0.0009264, 0.0027198, -0.0034724, 0.0034686
4: -0.0054574, -0.0016514, -0.0052341, -0.0018709, -0.0035865, 0.0035827
5: 0.0067719, 0.0108905, 0.0070135, 0.0106530, -0.0034611, 0.0034575
6: 0.0080691, 0.0103774, 0.0083951, 0.0102862, -0.0022171, 0.0019822
7: -0.0220415, -0.0131005, -0.0215259, -0.0136250, -0.0070024, 0.0070315
8: 0.9606394, 0.9862564, 0.9621167, 0.9847537, -0.0219925, 0.0220320
9: 0.0016924, 0.0092213, 0.0021340, 0.0087871, -0.0060595, 0.0060469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
time: 0.95 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004528, 0.0007713, -0.0011458, 0.0011011
1: -0.0011087, 0.0025076, -0.0007994, 0.0025036, -0.0035263, 0.0033070
2: 0.0125846, 0.0180004, 0.0125905, 0.0175372, -0.0047669, 0.0051190
3: -0.0011638, 0.0029086, -0.0011593, 0.0025603, -0.0035061, 0.0037806
4: -0.0054531, -0.0016967, -0.0054490, -0.0020180, -0.0034351, 0.0037523
5: 0.0067765, 0.0108415, 0.0067809, 0.0104939, -0.0034925, 0.0037673
6: 0.0080754, 0.0103756, 0.0080814, 0.0103740, -0.0022986, 0.0022943
7: -0.0219351, -0.0131106, -0.0211804, -0.0131202, -0.0075778, 0.0069571
8: 0.9609442, 0.9862276, 0.9631065, 0.9862001, -0.0240588, 0.0224480
9: 0.0017009, 0.0091317, 0.0017089, 0.0084962, -0.0060265, 0.0065445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144675
time: 0.97 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145069
time: 1.04 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005289, 0.0007937, -0.0004804, 0.0007054, -0.0010738, 0.0011279
1: -0.0011555, 0.0025381, -0.0009287, 0.0024027, -0.0033091, 0.0032876
2: 0.0125389, 0.0180705, 0.0127416, 0.0177307, -0.0047746, 0.0048049
3: -0.0011981, 0.0029614, -0.0010457, 0.0027059, -0.0035317, 0.0035530
4: -0.0054848, -0.0016481, -0.0053442, -0.0018837, -0.0036011, 0.0036962
5: 0.0067422, 0.0108941, 0.0068943, 0.0106391, -0.0035198, 0.0035412
6: 0.0080291, 0.0103886, 0.0082344, 0.0103312, -0.0023020, 0.0021542
7: -0.0220494, -0.0130361, -0.0214958, -0.0133663, -0.0071499, 0.0070978
8: 0.9606167, 0.9864410, 0.9622030, 0.9854949, -0.0225781, 0.0224324
9: 0.0016381, 0.0092279, 0.0019162, 0.0087617, -0.0061370, 0.0061802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145044
time: 0.95 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005195, 0.0007911, -0.0004513, 0.0008463, -0.0012102, 0.0011202
1: -0.0011119, 0.0025339, -0.0007925, 0.0026186, -0.0036280, 0.0033264
2: 0.0125451, 0.0180052, 0.0124184, 0.0175269, -0.0048583, 0.0052508
3: -0.0011935, 0.0029122, -0.0012888, 0.0025526, -0.0035700, 0.0038728
4: -0.0054805, -0.0016934, -0.0055684, -0.0020252, -0.0034554, 0.0038751
5: 0.0067469, 0.0108451, 0.0066517, 0.0104861, -0.0035563, 0.0038587
6: 0.0080354, 0.0103868, 0.0079070, 0.0104227, -0.0023873, 0.0024798
7: -0.0219429, -0.0130462, -0.0211635, -0.0128396, -0.0077310, 0.0070518
8: 0.9609217, 0.9864120, 0.9631548, 0.9870039, -0.0246928, 0.0228859
9: 0.0016466, 0.0091383, 0.0014727, 0.0084820, -0.0061206, 0.0066827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0142999
time: 1.08 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145070
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005179, 0.0007172, -0.0005117, 0.0006618, -0.0010307, 0.0010886
1: -0.0011043, 0.0024207, -0.0010750, 0.0023358, -0.0031660, 0.0032777
2: 0.0127147, 0.0179937, 0.0128418, 0.0179499, -0.0047802, 0.0046229
3: -0.0010660, 0.0029036, -0.0009704, 0.0028707, -0.0035443, 0.0034283
4: -0.0053629, -0.0017013, -0.0052747, -0.0017317, -0.0036312, 0.0035734
5: 0.0068741, 0.0108365, 0.0069695, 0.0108036, -0.0035333, 0.0034175
6: 0.0082071, 0.0103388, 0.0083358, 0.0103028, -0.0020957, 0.0020030
7: -0.0219243, -0.0133224, -0.0218529, -0.0135296, -0.0069537, 0.0072182
8: 0.9609751, 0.9856206, 0.9611799, 0.9850270, -0.0216962, 0.0224378
9: 0.0018792, 0.0091226, 0.0020537, 0.0090624, -0.0062091, 0.0059956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148999, upper bound: 0.0146061
time: 0.94 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
time: 0.87 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005179, 0.0007172, -0.0004997, 0.0006953, -0.0010692, 0.0010812
1: -0.0011043, 0.0024207, -0.0010189, 0.0023872, -0.0032805, 0.0033160
2: 0.0127147, 0.0179937, 0.0127650, 0.0178660, -0.0048165, 0.0047843
3: -0.0010660, 0.0029036, -0.0010282, 0.0028076, -0.0035630, 0.0035451
4: -0.0053629, -0.0017013, -0.0053280, -0.0017899, -0.0035730, 0.0036267
5: 0.0068741, 0.0108365, 0.0069119, 0.0107406, -0.0035512, 0.0035339
6: 0.0082071, 0.0103388, 0.0082580, 0.0103246, -0.0021175, 0.0020808
7: -0.0219243, -0.0133224, -0.0217161, -0.0134044, -0.0071724, 0.0071857
8: 0.9609751, 0.9856206, 0.9615716, 0.9853858, -0.0224623, 0.0226307
9: 0.0018792, 0.0091226, 0.0019482, 0.0089473, -0.0061996, 0.0061890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149734, upper bound: 0.0146768
time: 1.05 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0146327
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005144, 0.0007847, -0.0005123, 0.0006789, -0.0010462, 0.0011384
1: -0.0010879, 0.0025242, -0.0010782, 0.0023621, -0.0032179, 0.0033458
2: 0.0125597, 0.0179693, 0.0128025, 0.0179547, -0.0048627, 0.0046836
3: -0.0011825, 0.0028852, -0.0010000, 0.0028743, -0.0035966, 0.0034701
4: -0.0054704, -0.0017183, -0.0053020, -0.0017284, -0.0037420, 0.0035837
5: 0.0067578, 0.0108182, 0.0069400, 0.0108072, -0.0035846, 0.0034589
6: 0.0080502, 0.0103827, 0.0082960, 0.0103139, -0.0022638, 0.0020867
7: -0.0218844, -0.0130699, -0.0218607, -0.0134655, -0.0070018, 0.0072847
8: 0.9610892, 0.9863440, 0.9611574, 0.9852107, -0.0219947, 0.0228495
9: 0.0016666, 0.0090890, 0.0019997, 0.0090690, -0.0062691, 0.0060497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148999, upper bound: 0.0147331
time: 1.07 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0145498
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005144, 0.0007847, -0.0005004, 0.0007120, -0.0010881, 0.0011326
1: -0.0010879, 0.0025242, -0.0010224, 0.0024128, -0.0033540, 0.0033854
2: 0.0125597, 0.0179693, 0.0127265, 0.0178711, -0.0049012, 0.0048762
3: -0.0011825, 0.0028852, -0.0010571, 0.0028114, -0.0036179, 0.0036095
4: -0.0054704, -0.0017183, -0.0053547, -0.0017864, -0.0036840, 0.0036364
5: 0.0067578, 0.0108182, 0.0068830, 0.0107445, -0.0036053, 0.0035978
6: 0.0080502, 0.0103827, 0.0082190, 0.0103355, -0.0022853, 0.0021636
7: -0.0218844, -0.0130699, -0.0217245, -0.0133417, -0.0072669, 0.0072498
8: 0.9610892, 0.9863440, 0.9615476, 0.9855654, -0.0229048, 0.0230514
9: 0.0016666, 0.0090890, 0.0018955, 0.0089543, -0.0062579, 0.0062812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149735, upper bound: 0.0148188
time: 1.03 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0147555
time: 0.91 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004830, 0.0006363, -0.0005282, 0.0007765, -0.0011113, 0.0010145
1: -0.0009410, 0.0022967, -0.0011523, 0.0025117, -0.0032216, 0.0032185
2: 0.0129004, 0.0177492, 0.0125785, 0.0180657, -0.0046822, 0.0046914
3: -0.0009264, 0.0027198, -0.0011684, 0.0029577, -0.0034686, 0.0034724
4: -0.0052341, -0.0018709, -0.0054574, -0.0016514, -0.0035827, 0.0035865
5: 0.0070135, 0.0106530, 0.0067719, 0.0108905, -0.0034575, 0.0034611
6: 0.0083951, 0.0102862, 0.0080691, 0.0103774, -0.0019822, 0.0022171
7: -0.0215259, -0.0136250, -0.0220415, -0.0131005, -0.0070315, 0.0070024
8: 0.9621167, 0.9847537, 0.9606394, 0.9862564, -0.0220320, 0.0219925
9: 0.0021340, 0.0087871, 0.0016924, 0.0092213, -0.0060469, 0.0060595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145069, upper bound: 0.0146978
time: 0.82 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145069, upper bound: 0.0146978
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004528, 0.0007713, -0.0005189, 0.0007738, -0.0011011, 0.0011458
1: -0.0007994, 0.0025036, -0.0011087, 0.0025076, -0.0033070, 0.0035263
2: 0.0125905, 0.0175372, 0.0125846, 0.0180004, -0.0051190, 0.0047669
3: -0.0011593, 0.0025603, -0.0011638, 0.0029086, -0.0037806, 0.0035061
4: -0.0054490, -0.0020180, -0.0054531, -0.0016967, -0.0037523, 0.0034351
5: 0.0067809, 0.0104939, 0.0067765, 0.0108415, -0.0037673, 0.0034925
6: 0.0080814, 0.0103740, 0.0080754, 0.0103756, -0.0022943, 0.0022986
7: -0.0211804, -0.0131202, -0.0219351, -0.0131106, -0.0069571, 0.0075778
8: 0.9631065, 0.9862001, 0.9609442, 0.9862276, -0.0224480, 0.0240588
9: 0.0017089, 0.0084962, 0.0017009, 0.0091317, -0.0065445, 0.0060265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B2_A1_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144675, upper bound: 0.0145318
time: 1.06 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144675, upper bound: 0.0146978
time: 1.01 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004804, 0.0007054, -0.0005289, 0.0007937, -0.0011279, 0.0010738
1: -0.0009287, 0.0024027, -0.0011555, 0.0025381, -0.0032876, 0.0033091
2: 0.0127416, 0.0177307, 0.0125389, 0.0180705, -0.0048049, 0.0047746
3: -0.0010457, 0.0027059, -0.0011981, 0.0029614, -0.0035530, 0.0035317
4: -0.0053442, -0.0018837, -0.0054848, -0.0016481, -0.0036962, 0.0036011
5: 0.0068943, 0.0106391, 0.0067422, 0.0108941, -0.0035412, 0.0035198
6: 0.0082344, 0.0103312, 0.0080291, 0.0103886, -0.0021542, 0.0023020
7: -0.0214958, -0.0133663, -0.0220494, -0.0130361, -0.0070978, 0.0071499
8: 0.9622030, 0.9854949, 0.9606167, 0.9864410, -0.0224324, 0.0225781
9: 0.0019162, 0.0087617, 0.0016381, 0.0092279, -0.0061802, 0.0061370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149598
time: 0.85 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149604
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004513, 0.0008463, -0.0005195, 0.0007911, -0.0011202, 0.0012102
1: -0.0007925, 0.0026186, -0.0011119, 0.0025339, -0.0033264, 0.0036280
2: 0.0124184, 0.0175269, 0.0125451, 0.0180052, -0.0052508, 0.0048583
3: -0.0012888, 0.0025526, -0.0011935, 0.0029122, -0.0038728, 0.0035700
4: -0.0055684, -0.0020252, -0.0054805, -0.0016934, -0.0038751, 0.0034554
5: 0.0066517, 0.0104861, 0.0067469, 0.0108451, -0.0038587, 0.0035563
6: 0.0079070, 0.0104227, 0.0080354, 0.0103868, -0.0024798, 0.0023873
7: -0.0211635, -0.0128396, -0.0219429, -0.0130462, -0.0070518, 0.0077310
8: 0.9631548, 0.9870039, 0.9609217, 0.9864120, -0.0228859, 0.0246928
9: 0.0014727, 0.0084820, 0.0016466, 0.0091383, -0.0066827, 0.0061206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142999, upper bound: 0.0148199
time: 0.84 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142999, upper bound: 0.0148265
time: 1.05 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005117, 0.0006618, -0.0005179, 0.0007172, -0.0010886, 0.0010307
1: -0.0010750, 0.0023358, -0.0011043, 0.0024207, -0.0032777, 0.0031660
2: 0.0128418, 0.0179499, 0.0127147, 0.0179937, -0.0046229, 0.0047802
3: -0.0009704, 0.0028707, -0.0010660, 0.0029036, -0.0034283, 0.0035443
4: -0.0052747, -0.0017317, -0.0053629, -0.0017013, -0.0035734, 0.0036312
5: 0.0069695, 0.0108036, 0.0068741, 0.0108365, -0.0034175, 0.0035333
6: 0.0083358, 0.0103028, 0.0082071, 0.0103388, -0.0020030, 0.0020957
7: -0.0218529, -0.0135296, -0.0219243, -0.0133224, -0.0072182, 0.0069537
8: 0.9611799, 0.9850270, 0.9609751, 0.9856206, -0.0224378, 0.0216962
9: 0.0020537, 0.0090624, 0.0018792, 0.0091226, -0.0059956, 0.0062091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0148999
time: 1.04 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0147568
time: 1.04 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004997, 0.0006953, -0.0005179, 0.0007172, -0.0010812, 0.0010692
1: -0.0010189, 0.0023872, -0.0011043, 0.0024207, -0.0033160, 0.0032805
2: 0.0127650, 0.0178660, 0.0127147, 0.0179937, -0.0047843, 0.0048165
3: -0.0010282, 0.0028076, -0.0010660, 0.0029036, -0.0035451, 0.0035630
4: -0.0053280, -0.0017899, -0.0053629, -0.0017013, -0.0036267, 0.0035730
5: 0.0069119, 0.0107406, 0.0068741, 0.0108365, -0.0035339, 0.0035512
6: 0.0082580, 0.0103246, 0.0082071, 0.0103388, -0.0020808, 0.0021175
7: -0.0217161, -0.0134044, -0.0219243, -0.0133224, -0.0071857, 0.0071724
8: 0.9615716, 0.9853858, 0.9609751, 0.9856206, -0.0226307, 0.0224623
9: 0.0019482, 0.0089473, 0.0018792, 0.0091226, -0.0061890, 0.0061996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144080, upper bound: 0.0151933
time: 1.04 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0150511
time: 1.03 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005123, 0.0006789, -0.0005144, 0.0007847, -0.0011384, 0.0010462
1: -0.0010782, 0.0023621, -0.0010879, 0.0025242, -0.0033458, 0.0032179
2: 0.0128025, 0.0179547, 0.0125597, 0.0179693, -0.0046836, 0.0048627
3: -0.0010000, 0.0028743, -0.0011825, 0.0028852, -0.0034701, 0.0035966
4: -0.0053020, -0.0017284, -0.0054704, -0.0017183, -0.0035837, 0.0037420
5: 0.0069400, 0.0108072, 0.0067578, 0.0108182, -0.0034589, 0.0035846
6: 0.0082960, 0.0103139, 0.0080502, 0.0103827, -0.0020867, 0.0022638
7: -0.0218607, -0.0134655, -0.0218844, -0.0130699, -0.0072847, 0.0070018
8: 0.9611574, 0.9852107, 0.9610892, 0.9863440, -0.0228495, 0.0219947
9: 0.0019997, 0.0090690, 0.0016666, 0.0090890, -0.0060497, 0.0062691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146648, upper bound: 0.0148999
time: 1.00 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144721, upper bound: 0.0147568
time: 1.08 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005004, 0.0007120, -0.0005144, 0.0007847, -0.0011326, 0.0010881
1: -0.0010224, 0.0024128, -0.0010879, 0.0025242, -0.0033854, 0.0033540
2: 0.0127265, 0.0178711, 0.0125597, 0.0179693, -0.0048762, 0.0049012
3: -0.0010571, 0.0028114, -0.0011825, 0.0028852, -0.0036095, 0.0036179
4: -0.0053547, -0.0017864, -0.0054704, -0.0017183, -0.0036364, 0.0036840
5: 0.0068830, 0.0107445, 0.0067578, 0.0108182, -0.0035978, 0.0036053
6: 0.0082190, 0.0103355, 0.0080502, 0.0103827, -0.0021636, 0.0022853
7: -0.0217245, -0.0133417, -0.0218844, -0.0130699, -0.0072498, 0.0072669
8: 0.9615476, 0.9855654, 0.9610892, 0.9863440, -0.0230514, 0.0229048
9: 0.0018955, 0.0089543, 0.0016666, 0.0090890, -0.0062812, 0.0062579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145901, upper bound: 0.0151933
time: 0.96 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144721, upper bound: 0.0150511
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005282, 0.0007765, -0.0004999, 0.0007508, -0.0010979, 0.0010912
1: -0.0011523, 0.0025117, -0.0010202, 0.0024723, -0.0032989, 0.0032019
2: 0.0125785, 0.0180657, 0.0126375, 0.0178679, -0.0046176, 0.0047690
3: -0.0011684, 0.0029577, -0.0011240, 0.0028090, -0.0034005, 0.0035188
4: -0.0054574, -0.0016514, -0.0054164, -0.0017886, -0.0036687, 0.0037650
5: 0.0067719, 0.0108905, 0.0068162, 0.0107420, -0.0033880, 0.0035064
6: 0.0080691, 0.0103774, 0.0081289, 0.0103607, -0.0022915, 0.0022484
7: -0.0220415, -0.0131005, -0.0217192, -0.0131967, -0.0070556, 0.0067646
8: 0.9606394, 0.9862564, 0.9615629, 0.9859808, -0.0224315, 0.0217316
9: 0.0016924, 0.0092213, 0.0017734, 0.0089498, -0.0058547, 0.0060965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
time: 0.95 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004694, 0.0008874, -0.0012193, 0.0010811
1: -0.0011087, 0.0025076, -0.0008773, 0.0026816, -0.0036102, 0.0033055
2: 0.0125846, 0.0180004, 0.0123240, 0.0176539, -0.0047090, 0.0052209
3: -0.0011638, 0.0029086, -0.0013598, 0.0026481, -0.0034424, 0.0038457
4: -0.0054531, -0.0016967, -0.0056339, -0.0019370, -0.0035161, 0.0039372
5: 0.0067765, 0.0108415, 0.0065808, 0.0105814, -0.0034272, 0.0038311
6: 0.0080754, 0.0103756, 0.0078114, 0.0104495, -0.0023741, 0.0025642
7: -0.0219351, -0.0131106, -0.0213706, -0.0126858, -0.0075526, 0.0066879
8: 0.9609442, 0.9862276, 0.9625617, 0.9874446, -0.0245611, 0.0222210
9: 0.0017009, 0.0091317, 0.0013432, 0.0086563, -0.0058201, 0.0065761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144675
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145069
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005289, 0.0007937, -0.0004967, 0.0008206, -0.0011532, 0.0011077
1: -0.0011555, 0.0025381, -0.0010048, 0.0025793, -0.0033854, 0.0032767
2: 0.0125389, 0.0180705, 0.0124772, 0.0178448, -0.0047131, 0.0048760
3: -0.0011981, 0.0029614, -0.0012445, 0.0027917, -0.0034643, 0.0035881
4: -0.0054848, -0.0016481, -0.0055276, -0.0018046, -0.0036802, 0.0038795
5: 0.0067422, 0.0108941, 0.0066959, 0.0107247, -0.0034508, 0.0035746
6: 0.0080291, 0.0103886, 0.0079666, 0.0104061, -0.0023769, 0.0024219
7: -0.0220494, -0.0130361, -0.0216816, -0.0129355, -0.0071448, 0.0068360
8: 0.9606167, 0.9864410, 0.9616704, 0.9867291, -0.0229543, 0.0221971
9: 0.0016381, 0.0092279, 0.0015535, 0.0089182, -0.0059409, 0.0061748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
time: 0.87 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
time: 0.82 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005195, 0.0007911, -0.0004670, 0.0009507, -0.0012705, 0.0011001
1: -0.0011119, 0.0025339, -0.0008660, 0.0027787, -0.0037118, 0.0033737
2: 0.0125451, 0.0180052, 0.0121787, 0.0176369, -0.0048036, 0.0053498
3: -0.0011935, 0.0029122, -0.0014691, 0.0026353, -0.0035098, 0.0039325
4: -0.0054805, -0.0016934, -0.0057347, -0.0019488, -0.0035317, 0.0040413
5: 0.0067469, 0.0108451, 0.0064718, 0.0105686, -0.0034940, 0.0039168
6: 0.0080354, 0.0103868, 0.0076643, 0.0104906, -0.0024552, 0.0027226
7: -0.0219429, -0.0130462, -0.0213428, -0.0124490, -0.0076490, 0.0067870
8: 0.9609217, 0.9864120, 0.9626412, 0.9881230, -0.0251861, 0.0226729
9: 0.0016466, 0.0091383, 0.0011438, 0.0086329, -0.0059230, 0.0066737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147231, upper bound: 0.0144721
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147231, upper bound: 0.0145070
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005179, 0.0007172, -0.0005282, 0.0007765, -0.0011084, 0.0010660
1: -0.0011043, 0.0024207, -0.0011523, 0.0025117, -0.0032288, 0.0032500
2: 0.0127147, 0.0179937, 0.0125785, 0.0180657, -0.0046957, 0.0046775
3: -0.0010660, 0.0029036, -0.0011684, 0.0029577, -0.0034636, 0.0034534
4: -0.0053629, -0.0017013, -0.0054574, -0.0016514, -0.0037115, 0.0037561
5: 0.0068741, 0.0108365, 0.0067719, 0.0108905, -0.0034513, 0.0034414
6: 0.0082071, 0.0103388, 0.0080691, 0.0103774, -0.0021703, 0.0022697
7: -0.0219243, -0.0133224, -0.0220415, -0.0131005, -0.0069486, 0.0069361
8: 0.9609751, 0.9856206, 0.9606394, 0.9862564, -0.0219931, 0.0220892
9: 0.0018792, 0.0091226, 0.0016924, 0.0092213, -0.0059959, 0.0059973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149734, upper bound: 0.0145218
time: 0.82 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
time: 1.03 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005144, 0.0007847, -0.0005289, 0.0007937, -0.0011260, 0.0011185
1: -0.0010879, 0.0025242, -0.0011555, 0.0025381, -0.0033005, 0.0033340
2: 0.0125597, 0.0179693, 0.0125389, 0.0180705, -0.0047990, 0.0047722
3: -0.0011825, 0.0028852, -0.0011981, 0.0029614, -0.0035301, 0.0035213
4: -0.0054704, -0.0017183, -0.0054848, -0.0016481, -0.0038223, 0.0037665
5: 0.0067578, 0.0108182, 0.0067422, 0.0108941, -0.0035168, 0.0035088
6: 0.0080502, 0.0103827, 0.0080291, 0.0103886, -0.0023384, 0.0023536
7: -0.0218844, -0.0130699, -0.0220494, -0.0130361, -0.0070357, 0.0070193
8: 0.9610892, 0.9863440, 0.9606167, 0.9864410, -0.0224438, 0.0225946
9: 0.0016666, 0.0090890, 0.0016381, 0.0092279, -0.0060691, 0.0060860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0149735, upper bound: 0.0146512
time: 1.06 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0145500
time: 0.84 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005179, 0.0007172, -0.0005156, 0.0008112, -0.0011478, 0.0010603
1: -0.0011043, 0.0024207, -0.0010934, 0.0025648, -0.0033507, 0.0032941
2: 0.0127147, 0.0179937, 0.0124989, 0.0179775, -0.0047408, 0.0048542
3: -0.0010660, 0.0029036, -0.0012283, 0.0028914, -0.0034867, 0.0035810
4: -0.0053629, -0.0017013, -0.0055126, -0.0017126, -0.0036503, 0.0038113
5: 0.0068741, 0.0108365, 0.0067121, 0.0108243, -0.0034734, 0.0035681
6: 0.0082071, 0.0103388, 0.0079885, 0.0103999, -0.0021929, 0.0023503
7: -0.0219243, -0.0133224, -0.0218978, -0.0129708, -0.0071659, 0.0069009
8: 0.9609751, 0.9856206, 0.9610512, 0.9866281, -0.0228245, 0.0223238
9: 0.0018792, 0.0091226, 0.0015831, 0.0091002, -0.0059867, 0.0061890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151795, upper bound: 0.0146786
time: 0.91 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146339
time: 1.19 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005144, 0.0007847, -0.0005163, 0.0008282, -0.0011711, 0.0011125
1: -0.0010879, 0.0025242, -0.0010967, 0.0025909, -0.0034389, 0.0033718
2: 0.0125597, 0.0179693, 0.0124599, 0.0179825, -0.0048332, 0.0049726
3: -0.0011825, 0.0028852, -0.0012576, 0.0028952, -0.0035476, 0.0036666
4: -0.0054704, -0.0017183, -0.0055396, -0.0017091, -0.0037613, 0.0038214
5: 0.0067578, 0.0108182, 0.0066829, 0.0108280, -0.0035333, 0.0036532
6: 0.0080502, 0.0103827, 0.0079490, 0.0104110, -0.0023608, 0.0024336
7: -0.0218844, -0.0130699, -0.0219059, -0.0129073, -0.0073181, 0.0069707
8: 0.9610892, 0.9863440, 0.9610278, 0.9868101, -0.0233887, 0.0227774
9: 0.0016666, 0.0090890, 0.0015297, 0.0091071, -0.0060502, 0.0063315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0151795, upper bound: 0.0148194
time: 0.87 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0147583
time: 1.03 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.19 seconds
IS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144456, upper bound: 0.0145313
IS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144456, upper bound: 0.0145313
IS_B1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0144941
IS_B1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0145313
IS_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145233
IS_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
IS_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144981, upper bound: 0.0144981
IS_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144981, upper bound: 0.0145323
IS_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
IS_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
IS_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0145747
IS_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0147744
IS_B1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146343, upper bound: 0.0147477
IS_B1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146343, upper bound: 0.0148799
IS_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145323, upper bound: 0.0145749
IS_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145323, upper bound: 0.0147744
IS_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
IS_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
IS_B1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144675
IS_B1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145069
IS_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145044
IS_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
IS_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0142999
IS_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145070
IS_B1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0148999, upper bound: 0.0146061
IS_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
IS_B1_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0149734, upper bound: 0.0146768
IS_B1_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0146327
IS_B1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0148999, upper bound: 0.0147331
IS_B1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0145498
IS_B1_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0149735, upper bound: 0.0148188
IS_B1_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0147555
IS_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145069, upper bound: 0.0146978
IS_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145069, upper bound: 0.0146978
IS_B2_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144675, upper bound: 0.0145318
IS_B2_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144675, upper bound: 0.0146978
IS_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149598
IS_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149604
IS_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0142999, upper bound: 0.0148199
IS_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0142999, upper bound: 0.0148265
IS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0148999
IS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0147568
IS_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144080, upper bound: 0.0151933
IS_B2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0150511
IS_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146648, upper bound: 0.0148999
IS_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144721, upper bound: 0.0147568
IS_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145901, upper bound: 0.0151933
IS_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0144721, upper bound: 0.0150511
IS_B2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
IS_B2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0146978, upper bound: 0.0145069
IS_B2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144675
IS_B2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145069
IS_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
IS_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0148265, upper bound: 0.0145070
IS_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147231, upper bound: 0.0144721
IS_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147231, upper bound: 0.0145070
IS_B2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0149734, upper bound: 0.0145218
IS_B2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
IS_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0149735, upper bound: 0.0146512
IS_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0145500
IS_B2_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0151795, upper bound: 0.0146786
IS_B2_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146339
IS_B2_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0151795, upper bound: 0.0148194
IS_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0147583

## BFS IS instance: IS_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004919, 0.0006571, -0.0004830, 0.0006363, -0.0009620, 0.0009745
1: -0.0009825, 0.0023287, -0.0009410, 0.0022967, -0.0029438, 0.0029402
2: 0.0128525, 0.0178115, 0.0129004, 0.0177492, -0.0042699, 0.0042692
3: -0.0009624, 0.0027666, -0.0009264, 0.0027198, -0.0031555, 0.0031570
4: -0.0052673, -0.0018278, -0.0052341, -0.0018709, -0.0033964, 0.0034064
5: 0.0069775, 0.0106997, 0.0070135, 0.0106530, -0.0031447, 0.0031464
6: 0.0083466, 0.0102998, 0.0083951, 0.0102862, -0.0019396, 0.0019046
7: -0.0216273, -0.0135469, -0.0215259, -0.0136250, -0.0063838, 0.0063448
8: 0.9618262, 0.9849774, 0.9621167, 0.9847537, -0.0200680, 0.0200646
9: 0.0020683, 0.0088725, 0.0021340, 0.0087871, -0.0054813, 0.0055034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0146862
time: 0.98 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0146894
time: 0.96 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0007995, -0.0004830, 0.0006363, -0.0009492, 0.0011228
1: -0.0008582, 0.0025470, -0.0009410, 0.0022967, -0.0029942, 0.0033406
2: 0.0125256, 0.0176253, 0.0129004, 0.0177492, -0.0048401, 0.0043108
3: -0.0012081, 0.0026266, -0.0009264, 0.0027198, -0.0035695, 0.0031697
4: -0.0054940, -0.0019569, -0.0052341, -0.0018709, -0.0036231, 0.0032772
5: 0.0067322, 0.0105600, 0.0070135, 0.0106530, -0.0035565, 0.0031574
6: 0.0080156, 0.0103923, 0.0083951, 0.0102862, -0.0022706, 0.0019972
7: -0.0213240, -0.0130144, -0.0215259, -0.0136250, -0.0062669, 0.0070762
8: 0.9626952, 0.9865031, 0.9621167, 0.9847537, -0.0202974, 0.0227606
9: 0.0016199, 0.0086171, 0.0021340, 0.0087871, -0.0061410, 0.0054454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145251
time: 0.81 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
time: 0.79 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005023, 0.0006591, -0.0004620, 0.0007002, -0.0010446, 0.0009726
1: -0.0010312, 0.0023318, -0.0008424, 0.0023947, -0.0032494, 0.0029754
2: 0.0128479, 0.0178844, 0.0127537, 0.0176015, -0.0042893, 0.0047189
3: -0.0009658, 0.0028214, -0.0010367, 0.0026087, -0.0031582, 0.0034841
4: -0.0052705, -0.0017772, -0.0053359, -0.0019734, -0.0032971, 0.0035587
5: 0.0069741, 0.0107544, 0.0069034, 0.0105421, -0.0031464, 0.0034718
6: 0.0083420, 0.0103011, 0.0082466, 0.0103278, -0.0019857, 0.0020545
7: -0.0217461, -0.0135396, -0.0212852, -0.0133860, -0.0069318, 0.0063020
8: 0.9614856, 0.9849985, 0.9628063, 0.9854385, -0.0221793, 0.0201874
9: 0.0020621, 0.0089725, 0.0019328, 0.0085844, -0.0054443, 0.0060115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0142317
time: 0.85 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0140612
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005023, 0.0006591, -0.0004517, 0.0007329, -0.0010761, 0.0009664
1: -0.0010312, 0.0023318, -0.0007944, 0.0024449, -0.0033004, 0.0029723
2: 0.0128479, 0.0178844, 0.0126785, 0.0175297, -0.0042696, 0.0047953
3: -0.0009658, 0.0028214, -0.0010932, 0.0025547, -0.0031394, 0.0035416
4: -0.0052705, -0.0017772, -0.0053880, -0.0020232, -0.0032473, 0.0036109
5: 0.0069741, 0.0107544, 0.0068469, 0.0104882, -0.0031274, 0.0035291
6: 0.0083420, 0.0103011, 0.0081704, 0.0103491, -0.0020070, 0.0021306
7: -0.0217461, -0.0135396, -0.0211681, -0.0132634, -0.0070562, 0.0062402
8: 0.9614856, 0.9849985, 0.9631418, 0.9857897, -0.0225358, 0.0201105
9: 0.0020621, 0.0089725, 0.0018296, 0.0084858, -0.0054014, 0.0061163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1_B1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0142643
time: 1.08 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B2_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0141114
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005109, 0.0005641, -0.0004804, 0.0007054, -0.0010515, 0.0008801
1: -0.0010715, 0.0021862, -0.0009287, 0.0024027, -0.0030889, 0.0028321
2: 0.0130659, 0.0179447, 0.0127416, 0.0177307, -0.0040925, 0.0045181
3: -0.0008019, 0.0028668, -0.0010457, 0.0027059, -0.0030188, 0.0033528
4: -0.0051193, -0.0017353, -0.0053442, -0.0018837, -0.0032355, 0.0035214
5: 0.0071378, 0.0107997, 0.0068943, 0.0106391, -0.0030078, 0.0033425
6: 0.0085628, 0.0102393, 0.0082344, 0.0103312, -0.0017684, 0.0020049
7: -0.0218444, -0.0138948, -0.0214958, -0.0133663, -0.0068619, 0.0059863
8: 0.9612041, 0.9839807, 0.9622030, 0.9854949, -0.0211983, 0.0192477
9: 0.0023612, 0.0090553, 0.0019162, 0.0087617, -0.0052011, 0.0058916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145233
time: 0.95 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145233
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005104, 0.0006362, -0.0004804, 0.0007054, -0.0010379, 0.0009430
1: -0.0010692, 0.0022966, -0.0009287, 0.0024027, -0.0030764, 0.0029212
2: 0.0129007, 0.0179412, 0.0127416, 0.0177307, -0.0042094, 0.0044758
3: -0.0009261, 0.0028642, -0.0010457, 0.0027059, -0.0030991, 0.0033114
4: -0.0052339, -0.0017377, -0.0053442, -0.0018837, -0.0033502, 0.0035772
5: 0.0070137, 0.0107971, 0.0068943, 0.0106391, -0.0030874, 0.0033006
6: 0.0083954, 0.0102861, 0.0082344, 0.0103312, -0.0019358, 0.0020518
7: -0.0218387, -0.0136255, -0.0214958, -0.0133663, -0.0067103, 0.0061400
8: 0.9612203, 0.9847524, 0.9622030, 0.9854949, -0.0210252, 0.0198110
9: 0.0021344, 0.0090505, 0.0019162, 0.0087617, -0.0053267, 0.0057752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
time: 0.95 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005030, 0.0006762, -0.0004641, 0.0007781, -0.0011076, 0.0009984
1: -0.0010345, 0.0023580, -0.0008525, 0.0025141, -0.0033511, 0.0030656
2: 0.0128086, 0.0178892, 0.0125748, 0.0176167, -0.0044126, 0.0048505
3: -0.0009954, 0.0028250, -0.0011711, 0.0026201, -0.0032470, 0.0035741
4: -0.0052978, -0.0017738, -0.0054599, -0.0019629, -0.0033349, 0.0036861
5: 0.0069446, 0.0107580, 0.0067692, 0.0105535, -0.0032349, 0.0035606
6: 0.0083022, 0.0103122, 0.0080655, 0.0103784, -0.0020762, 0.0022467
7: -0.0217540, -0.0134755, -0.0213099, -0.0130946, -0.0070593, 0.0064753
8: 0.9614632, 0.9851820, 0.9627355, 0.9862733, -0.0228145, 0.0207777
9: 0.0020081, 0.0089791, 0.0016874, 0.0086052, -0.0055980, 0.0061316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0143195
time: 1.05 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0144981
time: 1.06 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005030, 0.0006762, -0.0004502, 0.0008016, -0.0011305, 0.0009865
1: -0.0010345, 0.0023580, -0.0007871, 0.0025501, -0.0033930, 0.0030411
2: 0.0128086, 0.0178892, 0.0125209, 0.0175188, -0.0043642, 0.0049132
3: -0.0009954, 0.0028250, -0.0012117, 0.0025465, -0.0032073, 0.0036213
4: -0.0052978, -0.0017738, -0.0054973, -0.0020307, -0.0032670, 0.0037235
5: 0.0069446, 0.0107580, 0.0067286, 0.0104800, -0.0031950, 0.0036077
6: 0.0083022, 0.0103122, 0.0080108, 0.0103937, -0.0020915, 0.0023014
7: -0.0217540, -0.0134755, -0.0211504, -0.0130066, -0.0071614, 0.0063478
8: 0.9614632, 0.9851820, 0.9631923, 0.9865254, -0.0231071, 0.0205592
9: 0.0020081, 0.0089791, 0.0016133, 0.0084709, -0.0055052, 0.0062176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0143346
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0145323
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004997, 0.0006953, -0.0004916, 0.0005596, -0.0008889, 0.0010171
1: -0.0010189, 0.0023872, -0.0009813, 0.0021792, -0.0029718, 0.0029525
2: 0.0127650, 0.0178660, 0.0130764, 0.0178095, -0.0043085, 0.0042863
3: -0.0010282, 0.0028076, -0.0007940, 0.0027651, -0.0031947, 0.0031523
4: -0.0053280, -0.0017899, -0.0051120, -0.0018291, -0.0033961, 0.0033221
5: 0.0069119, 0.0107406, 0.0071456, 0.0106983, -0.0031848, 0.0031400
6: 0.0082580, 0.0103246, 0.0085734, 0.0102363, -0.0019783, 0.0017511
7: -0.0217161, -0.0134044, -0.0216241, -0.0139119, -0.0061887, 0.0065336
8: 0.9615716, 0.9853858, 0.9618351, 0.9839317, -0.0201759, 0.0202256
9: 0.0019482, 0.0089473, 0.0023756, 0.0088698, -0.0056087, 0.0053866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
time: 0.78 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
time: 0.81 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004997, 0.0006953, -0.0004819, 0.0005993, -0.0009421, 0.0010134
1: -0.0010189, 0.0023872, -0.0009355, 0.0022401, -0.0030196, 0.0029976
2: 0.0127650, 0.0178660, 0.0129853, 0.0177411, -0.0043559, 0.0043727
3: -0.0010282, 0.0028076, -0.0008625, 0.0027136, -0.0032215, 0.0032293
4: -0.0053280, -0.0017899, -0.0051752, -0.0018766, -0.0034515, 0.0033853
5: 0.0069119, 0.0107406, 0.0070772, 0.0106468, -0.0032107, 0.0032181
6: 0.0082580, 0.0103246, 0.0084811, 0.0102622, -0.0020042, 0.0018434
7: -0.0217161, -0.0134044, -0.0215126, -0.0137633, -0.0064625, 0.0065116
8: 0.9615716, 0.9853858, 0.9621548, 0.9843574, -0.0205588, 0.0204640
9: 0.0019482, 0.0089473, 0.0022505, 0.0087759, -0.0056090, 0.0055907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
time: 1.00 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004897, 0.0006926, -0.0004620, 0.0007002, -0.0010343, 0.0010059
1: -0.0009724, 0.0023831, -0.0008424, 0.0023947, -0.0032282, 0.0030348
2: 0.0127710, 0.0177962, 0.0127537, 0.0176015, -0.0043781, 0.0046699
3: -0.0010236, 0.0027551, -0.0010367, 0.0026087, -0.0032250, 0.0034415
4: -0.0053238, -0.0018383, -0.0053359, -0.0019734, -0.0033505, 0.0034975
5: 0.0069164, 0.0106882, 0.0069034, 0.0105421, -0.0032130, 0.0034287
6: 0.0082641, 0.0103228, 0.0082466, 0.0103278, -0.0020636, 0.0020763
7: -0.0216024, -0.0134142, -0.0212852, -0.0133860, -0.0068135, 0.0064467
8: 0.9618973, 0.9853576, 0.9628063, 0.9854385, -0.0219676, 0.0206021
9: 0.0019566, 0.0088515, 0.0019328, 0.0085844, -0.0055662, 0.0059129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0143094
time: 0.86 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0141282
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004897, 0.0006926, -0.0004517, 0.0007329, -0.0010698, 0.0010023
1: -0.0009724, 0.0023831, -0.0007944, 0.0024449, -0.0033368, 0.0030823
2: 0.0127710, 0.0177962, 0.0126785, 0.0175297, -0.0044305, 0.0048352
3: -0.0010236, 0.0027551, -0.0010932, 0.0025547, -0.0032552, 0.0035649
4: -0.0053238, -0.0018383, -0.0053880, -0.0020232, -0.0033006, 0.0035497
5: 0.0069164, 0.0106882, 0.0068469, 0.0104882, -0.0032423, 0.0035516
6: 0.0082641, 0.0103228, 0.0081704, 0.0103491, -0.0020849, 0.0021524
7: -0.0216024, -0.0134142, -0.0211681, -0.0132634, -0.0070346, 0.0064272
8: 0.9618973, 0.9853576, 0.9631418, 0.9857897, -0.0227374, 0.0208678
9: 0.0019566, 0.0088515, 0.0018296, 0.0084858, -0.0055689, 0.0061197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0145156
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0143839
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005004, 0.0007120, -0.0004907, 0.0006316, -0.0009577, 0.0010390
1: -0.0010224, 0.0024128, -0.0009770, 0.0022896, -0.0029826, 0.0030370
2: 0.0127265, 0.0178711, 0.0129111, 0.0178031, -0.0044239, 0.0043058
3: -0.0010571, 0.0028114, -0.0009183, 0.0027603, -0.0032766, 0.0031763
4: -0.0053547, -0.0017864, -0.0052267, -0.0018335, -0.0035189, 0.0034403
5: 0.0068830, 0.0107445, 0.0070215, 0.0106934, -0.0032660, 0.0031650
6: 0.0082190, 0.0103355, 0.0084060, 0.0102832, -0.0020641, 0.0019295
7: -0.0217245, -0.0133417, -0.0216137, -0.0136425, -0.0063399, 0.0066668
8: 0.9615476, 0.9855654, 0.9618650, 0.9847037, -0.0202577, 0.0207754
9: 0.0018955, 0.0089543, 0.0021487, 0.0088610, -0.0057362, 0.0054915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0146264
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147477
time: 1.03 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005004, 0.0007120, -0.0004792, 0.0006651, -0.0009956, 0.0010355
1: -0.0010224, 0.0024128, -0.0009232, 0.0023410, -0.0030970, 0.0030823
2: 0.0127265, 0.0178711, 0.0128341, 0.0177226, -0.0044709, 0.0044694
3: -0.0010571, 0.0028114, -0.0009762, 0.0026997, -0.0033029, 0.0032932
4: -0.0053547, -0.0017864, -0.0052801, -0.0018894, -0.0034653, 0.0034937
5: 0.0068830, 0.0107445, 0.0069638, 0.0106330, -0.0032917, 0.0032812
6: 0.0082190, 0.0103355, 0.0083280, 0.0103050, -0.0020859, 0.0020074
7: -0.0217245, -0.0133417, -0.0214825, -0.0135171, -0.0065461, 0.0066486
8: 0.9615476, 0.9855654, 0.9622412, 0.9850628, -0.0210352, 0.0210133
9: 0.0018955, 0.0089543, 0.0020432, 0.0087505, -0.0057381, 0.0056653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147475
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004905, 0.0007094, -0.0004641, 0.0007781, -0.0011013, 0.0010310
1: -0.0009758, 0.0024088, -0.0008525, 0.0025141, -0.0033504, 0.0031231
2: 0.0127325, 0.0178013, 0.0125748, 0.0176167, -0.0044987, 0.0048357
3: -0.0010526, 0.0027589, -0.0011711, 0.0026201, -0.0033118, 0.0035582
4: -0.0053505, -0.0018348, -0.0054599, -0.0019629, -0.0033877, 0.0036251
5: 0.0068875, 0.0106921, 0.0067692, 0.0105535, -0.0032995, 0.0035445
6: 0.0082252, 0.0103337, 0.0080655, 0.0103784, -0.0021532, 0.0022683
7: -0.0216107, -0.0133515, -0.0213099, -0.0130946, -0.0069903, 0.0066155
8: 0.9618737, 0.9855372, 0.9627355, 0.9862733, -0.0227615, 0.0211796
9: 0.0019038, 0.0088585, 0.0016874, 0.0086052, -0.0057162, 0.0060791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0144456
time: 1.04 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0145749
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004905, 0.0007094, -0.0004502, 0.0008016, -0.0011271, 0.0010276
1: -0.0009758, 0.0024088, -0.0007871, 0.0025501, -0.0034347, 0.0031640
2: 0.0127325, 0.0178013, 0.0125209, 0.0175188, -0.0045460, 0.0049618
3: -0.0010526, 0.0027589, -0.0012117, 0.0025465, -0.0033382, 0.0036528
4: -0.0053505, -0.0018348, -0.0054973, -0.0020307, -0.0033198, 0.0036625
5: 0.0068875, 0.0106921, 0.0067286, 0.0104800, -0.0033249, 0.0036387
6: 0.0082252, 0.0103337, 0.0080108, 0.0103937, -0.0021685, 0.0023229
7: -0.0216107, -0.0133515, -0.0211504, -0.0130066, -0.0071534, 0.0065991
8: 0.9618737, 0.9855372, 0.9631923, 0.9865254, -0.0233470, 0.0214151
9: 0.0019038, 0.0088585, 0.0016133, 0.0084709, -0.0057187, 0.0062303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0146501
time: 1.04 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0147744
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005087, 0.0007719, -0.0004830, 0.0006363, -0.0009947, 0.0011065
1: -0.0010613, 0.0025046, -0.0009410, 0.0022967, -0.0031247, 0.0032142
2: 0.0125891, 0.0179294, 0.0129004, 0.0177492, -0.0046803, 0.0045400
3: -0.0011604, 0.0028553, -0.0009264, 0.0027198, -0.0034640, 0.0033581
4: -0.0054500, -0.0017459, -0.0052341, -0.0018709, -0.0035791, 0.0034882
5: 0.0067799, 0.0107882, 0.0070135, 0.0106530, -0.0034527, 0.0033470
6: 0.0080800, 0.0103744, 0.0083951, 0.0102862, -0.0022062, 0.0019792
7: -0.0218195, -0.0131179, -0.0215259, -0.0136250, -0.0067738, 0.0070133
8: 0.9612754, 0.9862067, 0.9621167, 0.9847537, -0.0213285, 0.0219801
9: 0.0017070, 0.0090343, 0.0021340, 0.0087871, -0.0060443, 0.0058489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146647
time: 0.88 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146675
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004804, 0.0009063, -0.0004830, 0.0006363, -0.0009766, 0.0012468
1: -0.0009288, 0.0027106, -0.0009410, 0.0022967, -0.0031409, 0.0034531
2: 0.0122806, 0.0177309, 0.0129004, 0.0177492, -0.0050380, 0.0045240
3: -0.0013924, 0.0027060, -0.0009264, 0.0027198, -0.0037330, 0.0033290
4: -0.0056640, -0.0018836, -0.0052341, -0.0018709, -0.0037931, 0.0033505
5: 0.0065483, 0.0106392, 0.0070135, 0.0106530, -0.0037213, 0.0033164
6: 0.0077675, 0.0104617, 0.0083951, 0.0102862, -0.0025187, 0.0020666
7: -0.0214960, -0.0126151, -0.0215259, -0.0136250, -0.0065790, 0.0075963
8: 0.9622023, 0.9876471, 0.9621167, 0.9847537, -0.0212997, 0.0236504
9: 0.0012837, 0.0087619, 0.0021340, 0.0087871, -0.0065352, 0.0057140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145060
time: 1.03 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004620, 0.0007002, -0.0010767, 0.0011047
1: -0.0011087, 0.0025076, -0.0008424, 0.0023947, -0.0034111, 0.0032496
2: 0.0125846, 0.0180004, 0.0127537, 0.0176015, -0.0046998, 0.0049464
3: -0.0011638, 0.0029086, -0.0010367, 0.0026087, -0.0034669, 0.0036508
4: -0.0054531, -0.0016967, -0.0053359, -0.0019734, -0.0034797, 0.0036392
5: 0.0067765, 0.0108415, 0.0069034, 0.0105421, -0.0034545, 0.0036378
6: 0.0080754, 0.0103756, 0.0082466, 0.0103278, -0.0022523, 0.0021291
7: -0.0219351, -0.0131106, -0.0212852, -0.0133860, -0.0072966, 0.0069709
8: 0.9609442, 0.9862276, 0.9628063, 0.9854385, -0.0232533, 0.0221038
9: 0.0017009, 0.0091317, 0.0019328, 0.0085844, -0.0060075, 0.0063078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142001
time: 0.95 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140289
time: 0.82 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004517, 0.0007329, -0.0011081, 0.0010984
1: -0.0011087, 0.0025076, -0.0007944, 0.0024449, -0.0034620, 0.0032465
2: 0.0125846, 0.0180004, 0.0126785, 0.0175297, -0.0046801, 0.0050228
3: -0.0011638, 0.0029086, -0.0010932, 0.0025547, -0.0034481, 0.0037082
4: -0.0054531, -0.0016967, -0.0053880, -0.0020232, -0.0034299, 0.0036913
5: 0.0067765, 0.0108415, 0.0068469, 0.0104882, -0.0034355, 0.0036951
6: 0.0080754, 0.0103756, 0.0081704, 0.0103491, -0.0022736, 0.0022052
7: -0.0219351, -0.0131106, -0.0211681, -0.0132634, -0.0074211, 0.0069091
8: 0.9609442, 0.9862276, 0.9631418, 0.9857897, -0.0236097, 0.0220270
9: 0.0017009, 0.0091317, 0.0018296, 0.0084858, -0.0059646, 0.0064125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142346
time: 0.97 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140810
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005275, 0.0006789, -0.0004804, 0.0007054, -0.0010865, 0.0010154
1: -0.0011494, 0.0023621, -0.0009287, 0.0024027, -0.0032822, 0.0031234
2: 0.0128024, 0.0180613, 0.0127416, 0.0177307, -0.0045287, 0.0048060
3: -0.0010000, 0.0029544, -0.0010457, 0.0027059, -0.0033468, 0.0035684
4: -0.0053020, -0.0016545, -0.0053442, -0.0018837, -0.0034183, 0.0036898
5: 0.0069400, 0.0108872, 0.0068943, 0.0106391, -0.0033352, 0.0035577
6: 0.0082960, 0.0103139, 0.0082344, 0.0103312, -0.0020352, 0.0020796
7: -0.0220344, -0.0134654, -0.0214958, -0.0133663, -0.0072797, 0.0066971
8: 0.9606598, 0.9852108, 0.9622030, 0.9854949, -0.0225453, 0.0212842
9: 0.0019997, 0.0092153, 0.0019162, 0.0087617, -0.0057996, 0.0062646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145044
time: 0.97 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145044
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005270, 0.0007501, -0.0004804, 0.0007054, -0.0010720, 0.0010743
1: -0.0011469, 0.0024712, -0.0009287, 0.0024027, -0.0032648, 0.0032032
2: 0.0126392, 0.0180575, 0.0127416, 0.0177307, -0.0046316, 0.0047571
3: -0.0011228, 0.0029516, -0.0010457, 0.0027059, -0.0034166, 0.0035232
4: -0.0054153, -0.0016571, -0.0053442, -0.0018837, -0.0035316, 0.0036872
5: 0.0068174, 0.0108844, 0.0068943, 0.0106391, -0.0034043, 0.0035118
6: 0.0081306, 0.0103602, 0.0082344, 0.0103312, -0.0022005, 0.0021258
7: -0.0220283, -0.0131994, -0.0214958, -0.0133663, -0.0071211, 0.0068280
8: 0.9606773, 0.9859732, 0.9622030, 0.9854949, -0.0223382, 0.0217822
9: 0.0017757, 0.0092101, 0.0019162, 0.0087617, -0.0059060, 0.0061461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
time: 1.03 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
time: 1.17 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005180, 0.0006762, -0.0004513, 0.0008463, -0.0012213, 0.0010077
1: -0.0011048, 0.0023580, -0.0007925, 0.0026186, -0.0035845, 0.0031505
2: 0.0128086, 0.0179946, 0.0124184, 0.0175269, -0.0046123, 0.0052251
3: -0.0009954, 0.0029043, -0.0012888, 0.0025526, -0.0033850, 0.0038690
4: -0.0052977, -0.0017007, -0.0055684, -0.0020252, -0.0032726, 0.0038677
5: 0.0069446, 0.0108372, 0.0066517, 0.0104861, -0.0033716, 0.0038564
6: 0.0083022, 0.0103122, 0.0079070, 0.0104227, -0.0021205, 0.0024052
7: -0.0219257, -0.0134755, -0.0211635, -0.0128396, -0.0078563, 0.0066510
8: 0.9609711, 0.9851819, 0.9631548, 0.9870039, -0.0245359, 0.0217375
9: 0.0020082, 0.0091238, 0.0014727, 0.0084820, -0.0057831, 0.0067564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142903
time: 1.09 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142999
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005177, 0.0007474, -0.0004513, 0.0008463, -0.0012084, 0.0010651
1: -0.0011034, 0.0024670, -0.0007925, 0.0026186, -0.0035823, 0.0032595
2: 0.0126454, 0.0179924, 0.0124184, 0.0175269, -0.0047343, 0.0051978
3: -0.0011181, 0.0029026, -0.0012888, 0.0025526, -0.0034643, 0.0038382
4: -0.0054110, -0.0017023, -0.0055684, -0.0020252, -0.0033858, 0.0038662
5: 0.0068221, 0.0108355, 0.0066517, 0.0104861, -0.0034497, 0.0038247
6: 0.0081370, 0.0103584, 0.0079070, 0.0104227, -0.0022858, 0.0024514
7: -0.0219221, -0.0132096, -0.0211635, -0.0128396, -0.0077039, 0.0067648
8: 0.9609815, 0.9859439, 0.9631548, 0.9870039, -0.0244327, 0.0223354
9: 0.0017842, 0.0091207, 0.0014727, 0.0084820, -0.0058867, 0.0066491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144721
time: 1.00 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145070
time: 1.12 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005179, 0.0007172, -0.0004919, 0.0006571, -0.0010259, 0.0010663
1: -0.0011043, 0.0024207, -0.0009825, 0.0023287, -0.0031588, 0.0031792
2: 0.0127147, 0.0179937, 0.0128525, 0.0178115, -0.0046217, 0.0046122
3: -0.0010660, 0.0029036, -0.0009624, 0.0027666, -0.0034220, 0.0034202
4: -0.0053629, -0.0017013, -0.0052673, -0.0018278, -0.0035352, 0.0035660
5: 0.0068741, 0.0108365, 0.0069775, 0.0106997, -0.0034110, 0.0034094
6: 0.0082071, 0.0103388, 0.0083466, 0.0102998, -0.0020927, 0.0019922
7: -0.0219243, -0.0133224, -0.0216273, -0.0135469, -0.0069361, 0.0069582
8: 0.9609751, 0.9856206, 0.9618262, 0.9849774, -0.0216459, 0.0217137
9: 0.0018792, 0.0091226, 0.0020683, 0.0088725, -0.0059871, 0.0059808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
time: 1.01 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
time: 1.00 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005081, 0.0007145, -0.0004653, 0.0007995, -0.0011668, 0.0010583
1: -0.0010582, 0.0024167, -0.0008582, 0.0025470, -0.0034850, 0.0032683
2: 0.0127207, 0.0179248, 0.0125256, 0.0176253, -0.0047027, 0.0050692
3: -0.0010614, 0.0028518, -0.0012081, 0.0026266, -0.0034577, 0.0037479
4: -0.0053587, -0.0017491, -0.0054940, -0.0019569, -0.0034018, 0.0037449
5: 0.0068787, 0.0107848, 0.0067322, 0.0105600, -0.0034444, 0.0037351
6: 0.0082132, 0.0103371, 0.0080156, 0.0103923, -0.0021791, 0.0023214
7: -0.0218119, -0.0133323, -0.0213240, -0.0130144, -0.0075569, 0.0069059
8: 0.9612970, 0.9855922, 0.9626952, 0.9865031, -0.0238147, 0.0221462
9: 0.0018876, 0.0090280, 0.0016199, 0.0086171, -0.0059661, 0.0065168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A1_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0144242
time: 1.00 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B2_B2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0144242
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004987, 0.0007125, -0.0004997, 0.0006953, -0.0010498, 0.0010763
1: -0.0010142, 0.0024136, -0.0010189, 0.0023872, -0.0031852, 0.0033085
2: 0.0127254, 0.0178589, 0.0127650, 0.0178660, -0.0048053, 0.0046426
3: -0.0010579, 0.0028022, -0.0010282, 0.0028076, -0.0035546, 0.0034401
4: -0.0053555, -0.0017949, -0.0053280, -0.0017899, -0.0035655, 0.0035332
5: 0.0068821, 0.0107353, 0.0069119, 0.0107406, -0.0035428, 0.0034292
6: 0.0082179, 0.0103358, 0.0082580, 0.0103246, -0.0021066, 0.0020778
7: -0.0217045, -0.0133399, -0.0217161, -0.0134044, -0.0069501, 0.0071674
8: 0.9616048, 0.9855705, 0.9615716, 0.9853858, -0.0217957, 0.0225784
9: 0.0018939, 0.0089375, 0.0019482, 0.0089473, -0.0061842, 0.0060036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146326
time: 0.88 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146327
time: 0.98 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004682, 0.0008471, -0.0004897, 0.0006926, -0.0010376, 0.0011995
1: -0.0008718, 0.0026199, -0.0009724, 0.0023831, -0.0032513, 0.0035765
2: 0.0124164, 0.0176455, 0.0127710, 0.0177962, -0.0051941, 0.0046857
3: -0.0012903, 0.0026418, -0.0010236, 0.0027551, -0.0038347, 0.0034523
4: -0.0055698, -0.0019428, -0.0053238, -0.0018383, -0.0037315, 0.0033810
5: 0.0066502, 0.0105752, 0.0069164, 0.0106882, -0.0038210, 0.0034396
6: 0.0079050, 0.0104233, 0.0082641, 0.0103228, -0.0024178, 0.0021591
7: -0.0213569, -0.0128364, -0.0216024, -0.0134142, -0.0068470, 0.0076193
8: 0.9626008, 0.9870131, 0.9618973, 0.9853576, -0.0220530, 0.0244129
9: 0.0014700, 0.0086448, 0.0019566, 0.0088515, -0.0066121, 0.0059483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A2_A1_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147923, upper bound: 0.0141613
time: 1.04 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146199, upper bound: 0.0141888
time: 0.96 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005144, 0.0007847, -0.0004926, 0.0006743, -0.0010415, 0.0011173
1: -0.0010879, 0.0025242, -0.0009857, 0.0023550, -0.0032107, 0.0032519
2: 0.0125597, 0.0179693, 0.0128131, 0.0178161, -0.0047116, 0.0046728
3: -0.0011825, 0.0028852, -0.0009920, 0.0027701, -0.0034801, 0.0034619
4: -0.0054704, -0.0017183, -0.0052946, -0.0018245, -0.0036459, 0.0035763
5: 0.0067578, 0.0108182, 0.0069480, 0.0107032, -0.0034681, 0.0034508
6: 0.0080502, 0.0103827, 0.0083068, 0.0103109, -0.0022608, 0.0020759
7: -0.0218844, -0.0130699, -0.0216349, -0.0134829, -0.0069842, 0.0070358
8: 0.9610892, 0.9863440, 0.9618043, 0.9851608, -0.0219443, 0.0221514
9: 0.0016666, 0.0090890, 0.0020144, 0.0088789, -0.0060532, 0.0060349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146952, upper bound: 0.0147331
time: 0.96 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146952, upper bound: 0.0147330
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005045, 0.0007820, -0.0004660, 0.0008175, -0.0011831, 0.0011077
1: -0.0010415, 0.0025201, -0.0008615, 0.0025744, -0.0035338, 0.0033574
2: 0.0125659, 0.0178997, 0.0124845, 0.0176301, -0.0048115, 0.0051260
3: -0.0011779, 0.0028329, -0.0012391, 0.0026302, -0.0035274, 0.0037873
4: -0.0054661, -0.0017666, -0.0055226, -0.0019535, -0.0035126, 0.0037560
5: 0.0067624, 0.0107659, 0.0067013, 0.0105636, -0.0035128, 0.0037742
6: 0.0080564, 0.0103810, 0.0079740, 0.0104040, -0.0023476, 0.0024070
7: -0.0217710, -0.0130799, -0.0213318, -0.0129473, -0.0076154, 0.0069686
8: 0.9614143, 0.9863154, 0.9626728, 0.9866953, -0.0240965, 0.0226791
9: 0.0016751, 0.0089935, 0.0015634, 0.0086236, -0.0060281, 0.0065723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145498
time: 0.98 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145498
time: 0.95 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004954, 0.0007800, -0.0005004, 0.0007120, -0.0010681, 0.0011277
1: -0.0009989, 0.0025171, -0.0010224, 0.0024128, -0.0032578, 0.0033780
2: 0.0125704, 0.0178360, 0.0127265, 0.0178711, -0.0048902, 0.0047298
3: -0.0011745, 0.0027850, -0.0010571, 0.0028114, -0.0036096, 0.0034991
4: -0.0054630, -0.0018108, -0.0053547, -0.0017864, -0.0036766, 0.0035440
5: 0.0067658, 0.0107181, 0.0068830, 0.0107445, -0.0035971, 0.0034874
6: 0.0080610, 0.0103797, 0.0082190, 0.0103355, -0.0022745, 0.0021606
7: -0.0216672, -0.0130874, -0.0217245, -0.0133417, -0.0070360, 0.0072318
8: 0.9617118, 0.9862940, 0.9615476, 0.9855654, -0.0222241, 0.0229999
9: 0.0016813, 0.0089061, 0.0018955, 0.0089543, -0.0062427, 0.0060818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148188
time: 0.84 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148188
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004658, 0.0009072, -0.0004905, 0.0007094, -0.0010574, 0.0012472
1: -0.0008602, 0.0027120, -0.0009758, 0.0024088, -0.0032691, 0.0036655
2: 0.0122785, 0.0176283, 0.0127325, 0.0178013, -0.0053075, 0.0047774
3: -0.0013940, 0.0026288, -0.0010526, 0.0027589, -0.0039127, 0.0035134
4: -0.0056655, -0.0019548, -0.0053505, -0.0018348, -0.0038307, 0.0033957
5: 0.0065467, 0.0105622, 0.0068875, 0.0106921, -0.0038981, 0.0034999
6: 0.0077653, 0.0104624, 0.0082252, 0.0103337, -0.0025684, 0.0022372
7: -0.0213288, -0.0126117, -0.0216107, -0.0133515, -0.0069488, 0.0077166
8: 0.9626812, 0.9876570, 0.9618737, 0.9855372, -0.0224934, 0.0249608
9: 0.0012807, 0.0086212, 0.0019038, 0.0088585, -0.0067046, 0.0060369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147555
time: 1.08 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147555
time: 1.02 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004830, 0.0006363, -0.0005087, 0.0007719, -0.0011065, 0.0009947
1: -0.0009410, 0.0022967, -0.0010613, 0.0025046, -0.0032142, 0.0031247
2: 0.0129004, 0.0177492, 0.0125891, 0.0179294, -0.0045400, 0.0046802
3: -0.0009264, 0.0027198, -0.0011604, 0.0028553, -0.0033581, 0.0034640
4: -0.0052341, -0.0018709, -0.0054500, -0.0017459, -0.0034882, 0.0035791
5: 0.0070135, 0.0106530, 0.0067799, 0.0107882, -0.0033470, 0.0034527
6: 0.0083951, 0.0102862, 0.0080800, 0.0103744, -0.0019792, 0.0022062
7: -0.0215259, -0.0136250, -0.0218195, -0.0131179, -0.0070133, 0.0067738
8: 0.9621167, 0.9847537, 0.9612754, 0.9862067, -0.0219801, 0.0213285
9: 0.0021340, 0.0087871, 0.0017070, 0.0090343, -0.0058489, 0.0060443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B2_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146647, upper bound: 0.0146896
time: 0.84 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146647, upper bound: 0.0148443
time: 1.00 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004830, 0.0006363, -0.0004804, 0.0009063, -0.0012468, 0.0009766
1: -0.0009410, 0.0022967, -0.0009288, 0.0027106, -0.0034531, 0.0031409
2: 0.0129004, 0.0177492, 0.0122806, 0.0177309, -0.0045240, 0.0050380
3: -0.0009264, 0.0027198, -0.0013924, 0.0027060, -0.0033290, 0.0037330
4: -0.0052341, -0.0018709, -0.0056640, -0.0018836, -0.0033505, 0.0037931
5: 0.0070135, 0.0106530, 0.0065483, 0.0106392, -0.0033164, 0.0037213
6: 0.0083951, 0.0102862, 0.0077675, 0.0104617, -0.0020666, 0.0025187
7: -0.0215259, -0.0136250, -0.0214960, -0.0126151, -0.0075963, 0.0065790
8: 0.9621167, 0.9847537, 0.9622023, 0.9876471, -0.0236504, 0.0212997
9: 0.0021340, 0.0087871, 0.0012837, 0.0087619, -0.0057140, 0.0065352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145060, upper bound: 0.0148443
time: 0.89 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_B2_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145060, upper bound: 0.0148443
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004620, 0.0007002, -0.0005189, 0.0007738, -0.0011047, 0.0010767
1: -0.0008424, 0.0023947, -0.0011087, 0.0025076, -0.0032496, 0.0034111
2: 0.0127537, 0.0176015, 0.0125846, 0.0180004, -0.0049464, 0.0046998
3: -0.0010367, 0.0026087, -0.0011638, 0.0029086, -0.0036508, 0.0034669
4: -0.0053359, -0.0019734, -0.0054531, -0.0016967, -0.0036392, 0.0034797
5: 0.0069034, 0.0105421, 0.0067765, 0.0108415, -0.0036378, 0.0034545
6: 0.0082466, 0.0103278, 0.0080754, 0.0103756, -0.0021291, 0.0022523
7: -0.0212852, -0.0133860, -0.0219351, -0.0131106, -0.0069709, 0.0072966
8: 0.9628063, 0.9854385, 0.9609442, 0.9862276, -0.0221038, 0.0232533
9: 0.0019328, 0.0085844, 0.0017009, 0.0091317, -0.0063078, 0.0060075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B2_A1_B1_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142001, upper bound: 0.0139994
time: 0.85 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140289, upper bound: 0.0140194
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004517, 0.0007329, -0.0005189, 0.0007738, -0.0010984, 0.0011081
1: -0.0007944, 0.0024449, -0.0011087, 0.0025076, -0.0032465, 0.0034620
2: 0.0126785, 0.0175297, 0.0125846, 0.0180004, -0.0050228, 0.0046801
3: -0.0010932, 0.0025547, -0.0011638, 0.0029086, -0.0037082, 0.0034481
4: -0.0053880, -0.0020232, -0.0054531, -0.0016967, -0.0036913, 0.0034299
5: 0.0068469, 0.0104882, 0.0067765, 0.0108415, -0.0036951, 0.0034355
6: 0.0081704, 0.0103491, 0.0080754, 0.0103756, -0.0022052, 0.0022736
7: -0.0211681, -0.0132634, -0.0219351, -0.0131106, -0.0069091, 0.0074211
8: 0.9631418, 0.9857897, 0.9609442, 0.9862276, -0.0220270, 0.0236097
9: 0.0018296, 0.0084858, 0.0017009, 0.0091317, -0.0064125, 0.0059646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B2_A1_B1_A1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142001, upper bound: 0.0141411
time: 1.06 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140289, upper bound: 0.0141693
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004804, 0.0007054, -0.0005275, 0.0006789, -0.0010154, 0.0010865
1: -0.0009287, 0.0024027, -0.0011494, 0.0023621, -0.0031234, 0.0032822
2: 0.0127416, 0.0177307, 0.0128024, 0.0180613, -0.0048060, 0.0045287
3: -0.0010457, 0.0027059, -0.0010000, 0.0029544, -0.0035684, 0.0033468
4: -0.0053442, -0.0018837, -0.0053020, -0.0016545, -0.0036898, 0.0034183
5: 0.0068943, 0.0106391, 0.0069400, 0.0108872, -0.0035577, 0.0033352
6: 0.0082344, 0.0103312, 0.0082960, 0.0103139, -0.0020796, 0.0020352
7: -0.0214958, -0.0133663, -0.0220344, -0.0134654, -0.0066971, 0.0072797
8: 0.9622030, 0.9854949, 0.9606598, 0.9852108, -0.0212842, 0.0225453
9: 0.0019162, 0.0087617, 0.0019997, 0.0092153, -0.0062646, 0.0057996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149598
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149597
time: 0.91 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004804, 0.0007054, -0.0005270, 0.0007501, -0.0010743, 0.0010720
1: -0.0009287, 0.0024027, -0.0011469, 0.0024712, -0.0032032, 0.0032648
2: 0.0127416, 0.0177307, 0.0126392, 0.0180575, -0.0047571, 0.0046316
3: -0.0010457, 0.0027059, -0.0011228, 0.0029516, -0.0035232, 0.0034166
4: -0.0053442, -0.0018837, -0.0054153, -0.0016571, -0.0036872, 0.0035316
5: 0.0068943, 0.0106391, 0.0068174, 0.0108844, -0.0035118, 0.0034043
6: 0.0082344, 0.0103312, 0.0081306, 0.0103602, -0.0021258, 0.0022005
7: -0.0214958, -0.0133663, -0.0220283, -0.0131994, -0.0068280, 0.0071211
8: 0.9622030, 0.9854949, 0.9606773, 0.9859732, -0.0217822, 0.0223382
9: 0.0019162, 0.0087617, 0.0017757, 0.0092101, -0.0061461, 0.0059060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149604
time: 1.08 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149604
time: 0.95 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004513, 0.0008463, -0.0005180, 0.0006762, -0.0010077, 0.0012213
1: -0.0007925, 0.0026186, -0.0011048, 0.0023580, -0.0031505, 0.0035845
2: 0.0124184, 0.0175269, 0.0128086, 0.0179946, -0.0052251, 0.0046123
3: -0.0012888, 0.0025526, -0.0009954, 0.0029043, -0.0038690, 0.0033850
4: -0.0055684, -0.0020252, -0.0052977, -0.0017007, -0.0038677, 0.0032726
5: 0.0066517, 0.0104861, 0.0069446, 0.0108372, -0.0038564, 0.0033716
6: 0.0079070, 0.0104227, 0.0083022, 0.0103122, -0.0024052, 0.0021205
7: -0.0211635, -0.0128396, -0.0219257, -0.0134755, -0.0066510, 0.0078563
8: 0.9631548, 0.9870039, 0.9609711, 0.9851819, -0.0217375, 0.0245359
9: 0.0014727, 0.0084820, 0.0020082, 0.0091238, -0.0067564, 0.0057831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B2_A1_B1_A2_A2_B1_A1

### Relational analysis result of IS_B2_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0147189
time: 0.92 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0148199
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004513, 0.0008463, -0.0005177, 0.0007474, -0.0010651, 0.0012084
1: -0.0007925, 0.0026186, -0.0011034, 0.0024670, -0.0032595, 0.0035823
2: 0.0124184, 0.0175269, 0.0126454, 0.0179924, -0.0051978, 0.0047343
3: -0.0012888, 0.0025526, -0.0011181, 0.0029026, -0.0038382, 0.0034643
4: -0.0055684, -0.0020252, -0.0054110, -0.0017023, -0.0038662, 0.0033858
5: 0.0066517, 0.0104861, 0.0068221, 0.0108355, -0.0038247, 0.0034497
6: 0.0079070, 0.0104227, 0.0081370, 0.0103584, -0.0024514, 0.0022858
7: -0.0211635, -0.0128396, -0.0219221, -0.0132096, -0.0067648, 0.0077039
8: 0.9631548, 0.9870039, 0.9609815, 0.9859439, -0.0223354, 0.0244327
9: 0.0014727, 0.0084820, 0.0017842, 0.0091207, -0.0066491, 0.0058867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of IS_B2_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0147231
time: 1.05 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0148265
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004919, 0.0006571, -0.0005179, 0.0007172, -0.0010663, 0.0010259
1: -0.0009825, 0.0023287, -0.0011043, 0.0024207, -0.0031792, 0.0031588
2: 0.0128525, 0.0178115, 0.0127147, 0.0179937, -0.0046122, 0.0046217
3: -0.0009624, 0.0027666, -0.0010660, 0.0029036, -0.0034202, 0.0034220
4: -0.0052673, -0.0018278, -0.0053629, -0.0017013, -0.0035660, 0.0035352
5: 0.0069775, 0.0106997, 0.0068741, 0.0108365, -0.0034094, 0.0034110
6: 0.0083466, 0.0102998, 0.0082071, 0.0103388, -0.0019922, 0.0020927
7: -0.0216273, -0.0135469, -0.0219243, -0.0133224, -0.0069582, 0.0069361
8: 0.9618262, 0.9849774, 0.9609751, 0.9856206, -0.0217137, 0.0216459
9: 0.0020683, 0.0088725, 0.0018792, 0.0091226, -0.0059808, 0.0059871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
time: 0.82 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004653, 0.0007995, -0.0005081, 0.0007145, -0.0010583, 0.0011668
1: -0.0008582, 0.0025470, -0.0010582, 0.0024167, -0.0032683, 0.0034850
2: 0.0125256, 0.0176253, 0.0127207, 0.0179248, -0.0050692, 0.0047027
3: -0.0012081, 0.0026266, -0.0010614, 0.0028518, -0.0037479, 0.0034577
4: -0.0054940, -0.0019569, -0.0053587, -0.0017491, -0.0037449, 0.0034018
5: 0.0067322, 0.0105600, 0.0068787, 0.0107848, -0.0037351, 0.0034444
6: 0.0080156, 0.0103923, 0.0082132, 0.0103371, -0.0023214, 0.0021791
7: -0.0213240, -0.0130144, -0.0218119, -0.0133323, -0.0069059, 0.0075569
8: 0.9626952, 0.9865031, 0.9612970, 0.9855922, -0.0221462, 0.0238147
9: 0.0016199, 0.0086171, 0.0018876, 0.0090280, -0.0065168, 0.0059661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0145482
time: 0.89 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
time: 0.83 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004997, 0.0006953, -0.0004987, 0.0007125, -0.0010763, 0.0010498
1: -0.0010189, 0.0023872, -0.0010142, 0.0024136, -0.0033085, 0.0031852
2: 0.0127650, 0.0178660, 0.0127254, 0.0178589, -0.0046426, 0.0048053
3: -0.0010282, 0.0028076, -0.0010579, 0.0028022, -0.0034401, 0.0035546
4: -0.0053280, -0.0017899, -0.0053555, -0.0017949, -0.0035332, 0.0035655
5: 0.0069119, 0.0107406, 0.0068821, 0.0107353, -0.0034292, 0.0035428
6: 0.0082580, 0.0103246, 0.0082179, 0.0103358, -0.0020778, 0.0021066
7: -0.0217161, -0.0134044, -0.0217045, -0.0133399, -0.0071674, 0.0069501
8: 0.9615716, 0.9853858, 0.9616048, 0.9855705, -0.0225784, 0.0217957
9: 0.0019482, 0.0089473, 0.0018939, 0.0089375, -0.0060036, 0.0061842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B2_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145857, upper bound: 0.0150511
time: 0.88 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145857, upper bound: 0.0150511
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004897, 0.0006926, -0.0004682, 0.0008471, -0.0011995, 0.0010376
1: -0.0009724, 0.0023831, -0.0008718, 0.0026199, -0.0035765, 0.0032513
2: 0.0127710, 0.0177962, 0.0124164, 0.0176455, -0.0046857, 0.0051941
3: -0.0010236, 0.0027551, -0.0012903, 0.0026418, -0.0034523, 0.0038347
4: -0.0053238, -0.0018383, -0.0055698, -0.0019428, -0.0033810, 0.0037315
5: 0.0069164, 0.0106882, 0.0066502, 0.0105752, -0.0034396, 0.0038210
6: 0.0082641, 0.0103228, 0.0079050, 0.0104233, -0.0021591, 0.0024178
7: -0.0216024, -0.0134142, -0.0213569, -0.0128364, -0.0076193, 0.0068470
8: 0.9618973, 0.9853576, 0.9626008, 0.9870131, -0.0244129, 0.0220530
9: 0.0019566, 0.0088515, 0.0014700, 0.0086448, -0.0059483, 0.0066121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B2_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141420, upper bound: 0.0147966
time: 1.04 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_B2_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141718, upper bound: 0.0146260
time: 0.87 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004926, 0.0006743, -0.0005144, 0.0007847, -0.0011173, 0.0010415
1: -0.0009857, 0.0023550, -0.0010879, 0.0025242, -0.0032519, 0.0032107
2: 0.0128131, 0.0178161, 0.0125597, 0.0179693, -0.0046728, 0.0047116
3: -0.0009920, 0.0027701, -0.0011825, 0.0028852, -0.0034619, 0.0034801
4: -0.0052946, -0.0018245, -0.0054704, -0.0017183, -0.0035763, 0.0036459
5: 0.0069480, 0.0107032, 0.0067578, 0.0108182, -0.0034508, 0.0034681
6: 0.0083068, 0.0103109, 0.0080502, 0.0103827, -0.0020759, 0.0022608
7: -0.0216349, -0.0134829, -0.0218844, -0.0130699, -0.0070358, 0.0069842
8: 0.9618043, 0.9851608, 0.9610892, 0.9863440, -0.0221514, 0.0219443
9: 0.0020144, 0.0088789, 0.0016666, 0.0090890, -0.0060349, 0.0060532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B2_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146061, upper bound: 0.0146952
time: 1.23 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146061, upper bound: 0.0148999
time: 1.07 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004660, 0.0008175, -0.0005045, 0.0007820, -0.0011077, 0.0011831
1: -0.0008615, 0.0025744, -0.0010415, 0.0025201, -0.0033574, 0.0035338
2: 0.0124845, 0.0176301, 0.0125659, 0.0178997, -0.0051260, 0.0048115
3: -0.0012391, 0.0026302, -0.0011779, 0.0028329, -0.0037873, 0.0035274
4: -0.0055226, -0.0019535, -0.0054661, -0.0017666, -0.0037560, 0.0035126
5: 0.0067013, 0.0105636, 0.0067624, 0.0107659, -0.0037742, 0.0035128
6: 0.0079740, 0.0104040, 0.0080564, 0.0103810, -0.0024070, 0.0023476
7: -0.0213318, -0.0129473, -0.0217710, -0.0130799, -0.0069686, 0.0076154
8: 0.9626728, 0.9866953, 0.9614143, 0.9863154, -0.0226791, 0.0240965
9: 0.0015634, 0.0086236, 0.0016751, 0.0089935, -0.0065723, 0.0060281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0145482
time: 1.04 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
time: 1.03 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005004, 0.0007120, -0.0004954, 0.0007800, -0.0011277, 0.0010681
1: -0.0010224, 0.0024128, -0.0009989, 0.0025171, -0.0033780, 0.0032578
2: 0.0127265, 0.0178711, 0.0125704, 0.0178360, -0.0047298, 0.0048902
3: -0.0010571, 0.0028114, -0.0011745, 0.0027850, -0.0034991, 0.0036096
4: -0.0053547, -0.0017864, -0.0054630, -0.0018108, -0.0035440, 0.0036766
5: 0.0068830, 0.0107445, 0.0067658, 0.0107181, -0.0034874, 0.0035971
6: 0.0082190, 0.0103355, 0.0080610, 0.0103797, -0.0021606, 0.0022745
7: -0.0217245, -0.0133417, -0.0216672, -0.0130874, -0.0072318, 0.0070360
8: 0.9615476, 0.9855654, 0.9617118, 0.9862940, -0.0229999, 0.0222241
9: 0.0018955, 0.0089543, 0.0016813, 0.0089061, -0.0060818, 0.0062427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146350, upper bound: 0.0150600
time: 1.11 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146350, upper bound: 0.0151933
time: 1.11 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004905, 0.0007094, -0.0004658, 0.0009072, -0.0012472, 0.0010574
1: -0.0009758, 0.0024088, -0.0008602, 0.0027120, -0.0036655, 0.0032691
2: 0.0127325, 0.0178013, 0.0122785, 0.0176283, -0.0047774, 0.0053075
3: -0.0010526, 0.0027589, -0.0013940, 0.0026288, -0.0035134, 0.0039127
4: -0.0053505, -0.0018348, -0.0056655, -0.0019548, -0.0033957, 0.0038307
5: 0.0068875, 0.0106921, 0.0065467, 0.0105622, -0.0034999, 0.0038981
6: 0.0082252, 0.0103337, 0.0077653, 0.0104624, -0.0022372, 0.0025684
7: -0.0216107, -0.0133515, -0.0213288, -0.0126117, -0.0077166, 0.0069488
8: 0.9618737, 0.9855372, 0.9626812, 0.9876570, -0.0249608, 0.0224934
9: 0.0019038, 0.0088585, 0.0012807, 0.0086212, -0.0060369, 0.0067046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145876, upper bound: 0.0149237
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145876, upper bound: 0.0150511
time: 1.04 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005087, 0.0007719, -0.0004999, 0.0007508, -0.0010760, 0.0010866
1: -0.0010613, 0.0025046, -0.0010202, 0.0024723, -0.0032052, 0.0031948
2: 0.0125891, 0.0179294, 0.0126375, 0.0178679, -0.0046069, 0.0046202
3: -0.0011604, 0.0028553, -0.0011240, 0.0028090, -0.0033925, 0.0034007
4: -0.0054500, -0.0017459, -0.0054164, -0.0017886, -0.0036613, 0.0036705
5: 0.0067799, 0.0107882, 0.0068162, 0.0107420, -0.0033799, 0.0033881
6: 0.0080800, 0.0103744, 0.0081289, 0.0103607, -0.0022807, 0.0022454
7: -0.0218195, -0.0131179, -0.0217192, -0.0131967, -0.0067968, 0.0067471
8: 0.9612754, 0.9862067, 0.9615629, 0.9859808, -0.0217453, 0.0216816
9: 0.0017070, 0.0090343, 0.0017734, 0.0089498, -0.0058400, 0.0058737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B2_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146650
time: 0.83 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146677
time: 1.01 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004804, 0.0009063, -0.0004999, 0.0007508, -0.0010623, 0.0012313
1: -0.0009288, 0.0027106, -0.0010202, 0.0024723, -0.0032467, 0.0034623
2: 0.0122806, 0.0177309, 0.0126375, 0.0178679, -0.0050075, 0.0046545
3: -0.0013924, 0.0027060, -0.0011240, 0.0028090, -0.0036937, 0.0034112
4: -0.0056640, -0.0018836, -0.0054164, -0.0017886, -0.0038753, 0.0035328
5: 0.0065483, 0.0106392, 0.0068162, 0.0107420, -0.0036806, 0.0033970
6: 0.0077675, 0.0104617, 0.0081289, 0.0103607, -0.0025931, 0.0023328
7: -0.0214960, -0.0126151, -0.0217192, -0.0131967, -0.0066684, 0.0073999
8: 0.9622023, 0.9876471, 0.9615629, 0.9859808, -0.0219389, 0.0235519
9: 0.0012837, 0.0087619, 0.0017734, 0.0089498, -0.0063897, 0.0058096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146650
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146677
time: 0.98 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004787, 0.0008152, -0.0011479, 0.0010845
1: -0.0011087, 0.0025076, -0.0009209, 0.0025710, -0.0034904, 0.0032381
2: 0.0125846, 0.0180004, 0.0124896, 0.0177192, -0.0046354, 0.0050415
3: -0.0011638, 0.0029086, -0.0012352, 0.0026972, -0.0033998, 0.0037108
4: -0.0054531, -0.0016967, -0.0055190, -0.0018918, -0.0035613, 0.0038223
5: 0.0067765, 0.0108415, 0.0067052, 0.0106304, -0.0033859, 0.0036964
6: 0.0080754, 0.0103756, 0.0079792, 0.0104025, -0.0023271, 0.0023965
7: -0.0219351, -0.0131106, -0.0214769, -0.0129557, -0.0072603, 0.0066977
8: 0.9609442, 0.9862276, 0.9622570, 0.9866712, -0.0237236, 0.0218530
9: 0.0017009, 0.0091317, 0.0015705, 0.0087458, -0.0058004, 0.0063299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B2_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142001
time: 0.87 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140289
time: 0.86 seconds

## BFS IS instance: IS_B2_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004682, 0.0008471, -0.0011799, 0.0010784
1: -0.0011087, 0.0025076, -0.0008718, 0.0026199, -0.0035455, 0.0032339
2: 0.0125846, 0.0180004, 0.0124164, 0.0176455, -0.0046208, 0.0051241
3: -0.0011638, 0.0029086, -0.0012903, 0.0026418, -0.0033826, 0.0037729
4: -0.0054531, -0.0016967, -0.0055698, -0.0019428, -0.0035103, 0.0038731
5: 0.0067765, 0.0108415, 0.0066502, 0.0105752, -0.0033683, 0.0037584
6: 0.0080754, 0.0103756, 0.0079050, 0.0104233, -0.0023479, 0.0024706
7: -0.0219351, -0.0131106, -0.0213569, -0.0128364, -0.0073948, 0.0066399
8: 0.9609442, 0.9862276, 0.9626008, 0.9870131, -0.0241090, 0.0217891
9: 0.0017009, 0.0091317, 0.0014700, 0.0086448, -0.0057573, 0.0064432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B2_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142346
time: 0.95 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140810
time: 1.05 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005094, 0.0007891, -0.0004967, 0.0008206, -0.0011320, 0.0011030
1: -0.0010644, 0.0025310, -0.0010048, 0.0025793, -0.0032954, 0.0032696
2: 0.0125496, 0.0179341, 0.0124772, 0.0178448, -0.0047025, 0.0047302
3: -0.0011901, 0.0028588, -0.0012445, 0.0027917, -0.0034563, 0.0034735
4: -0.0054774, -0.0017427, -0.0055276, -0.0018046, -0.0036728, 0.0037849
5: 0.0067502, 0.0107917, 0.0066959, 0.0107247, -0.0034429, 0.0034599
6: 0.0080399, 0.0103856, 0.0079666, 0.0104061, -0.0023661, 0.0024189
7: -0.0218271, -0.0130535, -0.0216816, -0.0129355, -0.0068951, 0.0068187
8: 0.9612536, 0.9863912, 0.9616704, 0.9867291, -0.0222817, 0.0221475
9: 0.0016528, 0.0090407, 0.0015535, 0.0089182, -0.0059263, 0.0059579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145045
time: 1.10 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146677
time: 1.08 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004811, 0.0009231, -0.0004967, 0.0008206, -0.0011182, 0.0012478
1: -0.0009319, 0.0027363, -0.0010048, 0.0025793, -0.0033490, 0.0035371
2: 0.0122420, 0.0177356, 0.0124772, 0.0178448, -0.0051031, 0.0047789
3: -0.0014214, 0.0027096, -0.0012445, 0.0027917, -0.0037575, 0.0034923
4: -0.0056908, -0.0018803, -0.0055276, -0.0018046, -0.0038861, 0.0036473
5: 0.0065193, 0.0106428, 0.0066959, 0.0107247, -0.0037435, 0.0034767
6: 0.0077284, 0.0104727, 0.0079666, 0.0104061, -0.0026776, 0.0025060
7: -0.0215037, -0.0125523, -0.0216816, -0.0129355, -0.0067772, 0.0074714
8: 0.9621801, 0.9878272, 0.9616704, 0.9867291, -0.0225486, 0.0240174
9: 0.0012307, 0.0087684, 0.0015535, 0.0089182, -0.0064758, 0.0059113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145045
time: 1.17 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146677
time: 1.20 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005195, 0.0007911, -0.0004793, 0.0008813, -0.0012025, 0.0011096
1: -0.0011119, 0.0025339, -0.0009236, 0.0026723, -0.0035991, 0.0033141
2: 0.0125451, 0.0180052, 0.0123380, 0.0177232, -0.0047480, 0.0051811
3: -0.0011935, 0.0029122, -0.0013492, 0.0027002, -0.0034818, 0.0038056
4: -0.0054805, -0.0016934, -0.0056242, -0.0018890, -0.0035915, 0.0039308
5: 0.0067469, 0.0108451, 0.0065914, 0.0106334, -0.0034677, 0.0037902
6: 0.0080354, 0.0103868, 0.0078256, 0.0104455, -0.0024101, 0.0025612
7: -0.0219429, -0.0130462, -0.0214834, -0.0127087, -0.0073741, 0.0068669
8: 0.9609217, 0.9864120, 0.9622383, 0.9873791, -0.0243985, 0.0223810
9: 0.0016466, 0.0091383, 0.0013624, 0.0087514, -0.0059502, 0.0064422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142903
time: 1.04 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144721
time: 0.95 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005195, 0.0007911, -0.0004658, 0.0009072, -0.0012281, 0.0010977
1: -0.0011119, 0.0025339, -0.0008602, 0.0027120, -0.0036454, 0.0033007
2: 0.0125451, 0.0180052, 0.0122785, 0.0176283, -0.0047118, 0.0052504
3: -0.0011935, 0.0029122, -0.0013940, 0.0026288, -0.0034491, 0.0038577
4: -0.0054805, -0.0016934, -0.0056655, -0.0019548, -0.0035257, 0.0039721
5: 0.0067469, 0.0108451, 0.0065467, 0.0105622, -0.0034342, 0.0038421
6: 0.0080354, 0.0103868, 0.0077653, 0.0104624, -0.0024270, 0.0026215
7: -0.0219429, -0.0130462, -0.0213288, -0.0126117, -0.0074870, 0.0067423
8: 0.9609217, 0.9864120, 0.9626812, 0.9876570, -0.0247218, 0.0222249
9: 0.0016466, 0.0091383, 0.0012807, 0.0086212, -0.0058607, 0.0065373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142999
time: 1.02 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145070
time: 1.03 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004987, 0.0007125, -0.0005282, 0.0007765, -0.0010883, 0.0010613
1: -0.0010142, 0.0024136, -0.0011523, 0.0025117, -0.0031369, 0.0032427
2: 0.0127254, 0.0178589, 0.0125785, 0.0180657, -0.0046848, 0.0045344
3: -0.0010579, 0.0028022, -0.0011684, 0.0029577, -0.0034554, 0.0033448
4: -0.0053555, -0.0017949, -0.0054574, -0.0016514, -0.0037041, 0.0036625
5: 0.0068821, 0.0107353, 0.0067719, 0.0108905, -0.0034431, 0.0033328
6: 0.0082179, 0.0103358, 0.0080691, 0.0103774, -0.0021595, 0.0022666
7: -0.0217045, -0.0133399, -0.0220415, -0.0131005, -0.0067155, 0.0069183
8: 0.9616048, 0.9855705, 0.9606394, 0.9862564, -0.0213273, 0.0220381
9: 0.0018939, 0.0089375, 0.0016924, 0.0092213, -0.0059809, 0.0057948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A2_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
time: 1.01 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
time: 1.02 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004682, 0.0008471, -0.0005189, 0.0007738, -0.0010784, 0.0011799
1: -0.0008718, 0.0026199, -0.0011087, 0.0025076, -0.0032339, 0.0035455
2: 0.0124164, 0.0176455, 0.0125846, 0.0180004, -0.0051241, 0.0046208
3: -0.0012903, 0.0026418, -0.0011638, 0.0029086, -0.0037729, 0.0033826
4: -0.0055698, -0.0019428, -0.0054531, -0.0016967, -0.0038731, 0.0035103
5: 0.0066502, 0.0105752, 0.0067765, 0.0108415, -0.0037584, 0.0033683
6: 0.0079050, 0.0104233, 0.0080754, 0.0103756, -0.0024706, 0.0023479
7: -0.0213569, -0.0128364, -0.0219351, -0.0131106, -0.0066399, 0.0073948
8: 0.9626008, 0.9870131, 0.9609442, 0.9862276, -0.0217891, 0.0241090
9: 0.0014700, 0.0086448, 0.0017009, 0.0091317, -0.0064432, 0.0057573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B2_A2_A2_B1_A1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144736, upper bound: 0.0139163
time: 0.99 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143116, upper bound: 0.0139482
time: 0.99 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004954, 0.0007800, -0.0005289, 0.0007937, -0.0011051, 0.0011137
1: -0.0009989, 0.0025171, -0.0011555, 0.0025381, -0.0032094, 0.0033268
2: 0.0125704, 0.0178360, 0.0125389, 0.0180705, -0.0047882, 0.0046283
3: -0.0011745, 0.0027850, -0.0011981, 0.0029614, -0.0035220, 0.0034075
4: -0.0054630, -0.0018108, -0.0054848, -0.0016481, -0.0038149, 0.0036740
5: 0.0067658, 0.0107181, 0.0067422, 0.0108941, -0.0035087, 0.0033949
6: 0.0080610, 0.0103797, 0.0080291, 0.0103886, -0.0023276, 0.0023505
7: -0.0216672, -0.0130874, -0.0220494, -0.0130361, -0.0067902, 0.0070017
8: 0.9617118, 0.9862940, 0.9606167, 0.9864410, -0.0217839, 0.0225443
9: 0.0016813, 0.0089061, 0.0016381, 0.0092279, -0.0060543, 0.0058793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A2_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147853, upper bound: 0.0146507
time: 0.88 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147853, upper bound: 0.0146512
time: 1.01 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004658, 0.0009072, -0.0005195, 0.0007911, -0.0010977, 0.0012281
1: -0.0008602, 0.0027120, -0.0011119, 0.0025339, -0.0033007, 0.0036454
2: 0.0122785, 0.0176283, 0.0125451, 0.0180052, -0.0052504, 0.0047118
3: -0.0013940, 0.0026288, -0.0011935, 0.0029122, -0.0038577, 0.0034491
4: -0.0056655, -0.0019548, -0.0054805, -0.0016934, -0.0039721, 0.0035257
5: 0.0065467, 0.0105622, 0.0067469, 0.0108451, -0.0038421, 0.0034342
6: 0.0077653, 0.0104624, 0.0080354, 0.0103868, -0.0026215, 0.0024270
7: -0.0213288, -0.0126117, -0.0219429, -0.0130462, -0.0067423, 0.0074870
8: 0.9626812, 0.9876570, 0.9609217, 0.9864120, -0.0222249, 0.0247218
9: 0.0012807, 0.0086212, 0.0016466, 0.0091383, -0.0065373, 0.0058607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A2_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145500
time: 1.08 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145500
time: 1.10 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004987, 0.0007125, -0.0005156, 0.0008112, -0.0011271, 0.0010556
1: -0.0010142, 0.0024136, -0.0010934, 0.0025648, -0.0032575, 0.0032871
2: 0.0127254, 0.0178589, 0.0124989, 0.0179775, -0.0047303, 0.0047037
3: -0.0010579, 0.0028022, -0.0012283, 0.0028914, -0.0034788, 0.0034656
4: -0.0053555, -0.0017949, -0.0055126, -0.0017126, -0.0036429, 0.0037177
5: 0.0068821, 0.0107353, 0.0067121, 0.0108243, -0.0034654, 0.0034527
6: 0.0082179, 0.0103358, 0.0079885, 0.0103999, -0.0021820, 0.0023472
7: -0.0217045, -0.0133399, -0.0218978, -0.0129708, -0.0069223, 0.0068837
8: 0.9616048, 0.9855705, 0.9610512, 0.9866281, -0.0221297, 0.0222745
9: 0.0018939, 0.0089375, 0.0015831, 0.0091002, -0.0059722, 0.0059772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146338
time: 1.07 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146339
time: 1.10 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004682, 0.0008471, -0.0005057, 0.0008085, -0.0011162, 0.0011761
1: -0.0008718, 0.0026199, -0.0010469, 0.0025608, -0.0033511, 0.0035933
2: 0.0124164, 0.0176455, 0.0125050, 0.0179079, -0.0051780, 0.0047892
3: -0.0012903, 0.0026418, -0.0012237, 0.0028391, -0.0038049, 0.0035043
4: -0.0055698, -0.0019428, -0.0055084, -0.0017609, -0.0038089, 0.0035655
5: 0.0066502, 0.0105752, 0.0067167, 0.0107721, -0.0037896, 0.0034892
6: 0.0079050, 0.0104233, 0.0079947, 0.0103982, -0.0024932, 0.0024286
7: -0.0213569, -0.0128364, -0.0217844, -0.0129807, -0.0068357, 0.0073873
8: 0.9626008, 0.9870131, 0.9613760, 0.9865996, -0.0225900, 0.0243782
9: 0.0014700, 0.0086448, 0.0015915, 0.0090048, -0.0064589, 0.0059378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147923, upper bound: 0.0141624
time: 1.10 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146199, upper bound: 0.0141936
time: 1.00 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004954, 0.0007800, -0.0005163, 0.0008282, -0.0011489, 0.0011078
1: -0.0009989, 0.0025171, -0.0010967, 0.0025909, -0.0033427, 0.0033647
2: 0.0125704, 0.0178360, 0.0124599, 0.0179825, -0.0048226, 0.0048211
3: -0.0011745, 0.0027850, -0.0012576, 0.0028952, -0.0035396, 0.0035488
4: -0.0054630, -0.0018108, -0.0055396, -0.0017091, -0.0037538, 0.0037289
5: 0.0067658, 0.0107181, 0.0066829, 0.0108280, -0.0035254, 0.0035352
6: 0.0080610, 0.0103797, 0.0079490, 0.0104110, -0.0023500, 0.0024306
7: -0.0216672, -0.0130874, -0.0219059, -0.0129073, -0.0070566, 0.0069534
8: 0.9617118, 0.9862940, 0.9610278, 0.9868101, -0.0226910, 0.0227278
9: 0.0016813, 0.0089061, 0.0015297, 0.0091071, -0.0060356, 0.0061052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148194
time: 0.87 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148194
time: 1.02 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004658, 0.0009072, -0.0005064, 0.0008255, -0.0011410, 0.0012246
1: -0.0008602, 0.0027120, -0.0010503, 0.0025868, -0.0034253, 0.0036916
2: 0.0122785, 0.0176283, 0.0124660, 0.0179130, -0.0053027, 0.0048966
3: -0.0013940, 0.0026288, -0.0012530, 0.0028429, -0.0038908, 0.0035841
4: -0.0056655, -0.0019548, -0.0055354, -0.0017574, -0.0039081, 0.0035806
5: 0.0065467, 0.0105622, 0.0066874, 0.0107759, -0.0038747, 0.0035688
6: 0.0077653, 0.0104624, 0.0079552, 0.0104092, -0.0026439, 0.0025071
7: -0.0213288, -0.0126117, -0.0217927, -0.0129172, -0.0070018, 0.0074835
8: 0.9626812, 0.9876570, 0.9613522, 0.9867816, -0.0230963, 0.0249806
9: 0.0012807, 0.0086212, 0.0015380, 0.0090117, -0.0065554, 0.0060828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147583
time: 1.07 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147583
time: 1.07 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.55 seconds
IS_B1_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0146862
IS_B1_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0146894
IS_B1_A1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145251
IS_B1_A1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
IS_B1_A1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0142317
IS_B1_A1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0140612
IS_B1_A1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0142643
IS_B1_A1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0141114
IS_B1_A1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145233
IS_B1_A1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0145233
IS_B1_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
IS_B1_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145479, upper bound: 0.0146894
IS_B1_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0143195
IS_B1_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0144981
IS_B1_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0143346
IS_B1_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143195, upper bound: 0.0145323
IS_B1_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
IS_B1_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
IS_B1_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
IS_B1_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
IS_B1_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0143094
IS_B1_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0141282
IS_B1_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0145156
IS_B1_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0143839
IS_B1_A1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0146264
IS_B1_A1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147477
IS_B1_A1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147475
IS_B1_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0148799
IS_B1_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0144456
IS_B1_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0145749
IS_B1_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0146501
IS_B1_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143346, upper bound: 0.0147744
IS_B1_A2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146647
IS_B1_A2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146675
IS_B1_A2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145060
IS_B1_A2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
IS_B1_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142001
IS_B1_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140289
IS_B1_A2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142346
IS_B1_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140810
IS_B1_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145044
IS_B1_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145044
IS_B1_A2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
IS_B1_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146675
IS_B1_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142903
IS_B1_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142999
IS_B1_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144721
IS_B1_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145070
IS_B1_A2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
IS_B1_A2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
IS_B1_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0144242
IS_B1_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0144242
IS_B1_A2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146326
IS_B1_A2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146327
IS_B1_A2_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147923, upper bound: 0.0141613
IS_B1_A2_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146199, upper bound: 0.0141888
IS_B1_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146952, upper bound: 0.0147331
IS_B1_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146952, upper bound: 0.0147330
IS_B1_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145498
IS_B1_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145498
IS_B1_A2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148188
IS_B1_A2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148188
IS_B1_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147555
IS_B1_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147555
IS_B2_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146647, upper bound: 0.0146896
IS_B2_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146647, upper bound: 0.0148443
IS_B2_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145060, upper bound: 0.0148443
IS_B2_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145060, upper bound: 0.0148443
IS_B2_A1_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0142001, upper bound: 0.0139994
IS_B2_A1_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0140289, upper bound: 0.0140194
IS_B2_A1_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0142001, upper bound: 0.0141411
IS_B2_A1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0140289, upper bound: 0.0141693
IS_B2_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149598
IS_B2_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149597
IS_B2_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149604
IS_B2_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145044, upper bound: 0.0149604
IS_B2_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0147189
IS_B2_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0148199
IS_B2_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0147231
IS_B2_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0142903, upper bound: 0.0148265
IS_B2_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
IS_B2_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
IS_B2_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0145482
IS_B2_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
IS_B2_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145857, upper bound: 0.0150511
IS_B2_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145857, upper bound: 0.0150511
IS_B2_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0141420, upper bound: 0.0147966
IS_B2_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0141718, upper bound: 0.0146260
IS_B2_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146061, upper bound: 0.0146952
IS_B2_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146061, upper bound: 0.0148999
IS_B2_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0145482
IS_B2_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144242, upper bound: 0.0147568
IS_B2_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146350, upper bound: 0.0150600
IS_B2_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146350, upper bound: 0.0151933
IS_B2_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145876, upper bound: 0.0149237
IS_B2_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145876, upper bound: 0.0150511
IS_B2_A2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146650
IS_B2_A2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146677
IS_B2_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146650
IS_B2_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146896, upper bound: 0.0146677
IS_B2_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142001
IS_B2_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140289
IS_B2_A2_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142346
IS_B2_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140810
IS_B2_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145045
IS_B2_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146677
IS_B2_A2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0145045
IS_B2_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148443, upper bound: 0.0146677
IS_B2_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142903
IS_B2_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0144721
IS_B2_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0142999
IS_B2_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145318, upper bound: 0.0145070
IS_B2_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
IS_B2_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147568, upper bound: 0.0144242
IS_B2_A2_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0144736, upper bound: 0.0139163
IS_B2_A2_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0143116, upper bound: 0.0139482
IS_B2_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147853, upper bound: 0.0146507
IS_B2_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147853, upper bound: 0.0146512
IS_B2_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145500
IS_B2_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0145482, upper bound: 0.0145500
IS_B2_A2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146338
IS_B2_A2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150436, upper bound: 0.0146339
IS_B2_A2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0147923, upper bound: 0.0141624
IS_B2_A2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0146199, upper bound: 0.0141936
IS_B2_A2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148194
IS_B2_A2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0150107, upper bound: 0.0148194
IS_B2_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147583
IS_B2_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.55
Output dim: 8, lower bound: -0.0148944, upper bound: 0.0147583

## BFS IS instance: IS_B1_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004919, 0.0006571, -0.0004916, 0.0005596, -0.0008756, 0.0009792
1: -0.0009825, 0.0023287, -0.0009813, 0.0021792, -0.0028916, 0.0028860
2: 0.0128525, 0.0178115, 0.0130764, 0.0178095, -0.0042090, 0.0041766
3: -0.0009624, 0.0027666, -0.0007940, 0.0027651, -0.0031199, 0.0030736
4: -0.0052673, -0.0018278, -0.0051120, -0.0018291, -0.0033271, 0.0032842
5: 0.0069775, 0.0106997, 0.0071456, 0.0106983, -0.0031100, 0.0030617
6: 0.0083466, 0.0102998, 0.0085734, 0.0102363, -0.0018897, 0.0017263
7: -0.0216273, -0.0135469, -0.0216241, -0.0139119, -0.0060315, 0.0063714
8: 0.9618262, 0.9849774, 0.9618351, 0.9839317, -0.0196555, 0.0197608
9: 0.0020683, 0.0088725, 0.0023756, 0.0088698, -0.0054721, 0.0052523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148937, upper bound: 0.0149098
time: 0.92 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148937, upper bound: 0.0150724
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004919, 0.0006571, -0.0004819, 0.0005993, -0.0009264, 0.0009715
1: -0.0009825, 0.0023287, -0.0009355, 0.0022401, -0.0028846, 0.0028775
2: 0.0128525, 0.0178115, 0.0129853, 0.0177411, -0.0041899, 0.0041805
3: -0.0009624, 0.0027666, -0.0008625, 0.0027136, -0.0031018, 0.0030903
4: -0.0052673, -0.0018278, -0.0051752, -0.0018766, -0.0033513, 0.0033475
5: 0.0069775, 0.0106997, 0.0070772, 0.0106468, -0.0030917, 0.0030798
6: 0.0083466, 0.0102998, 0.0084811, 0.0102622, -0.0019155, 0.0018187
7: -0.0216273, -0.0135469, -0.0215126, -0.0137633, -0.0062393, 0.0062962
8: 0.9618262, 0.9849774, 0.9621548, 0.9843574, -0.0196539, 0.0196786
9: 0.0020683, 0.0088725, 0.0022505, 0.0087759, -0.0054219, 0.0053817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148937, upper bound: 0.0149098
time: 1.10 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148937, upper bound: 0.0150724
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004620, 0.0007002, -0.0004830, 0.0006363, -0.0009416, 0.0010237
1: -0.0008424, 0.0023947, -0.0009410, 0.0022967, -0.0029026, 0.0031862
2: 0.0127537, 0.0176015, 0.0129004, 0.0177492, -0.0046088, 0.0041981
3: -0.0010367, 0.0026087, -0.0009264, 0.0027198, -0.0033956, 0.0030951
4: -0.0053359, -0.0019734, -0.0052341, -0.0018709, -0.0034650, 0.0032607
5: 0.0069034, 0.0105421, 0.0070135, 0.0106530, -0.0033829, 0.0030838
6: 0.0082466, 0.0103278, 0.0083951, 0.0102862, -0.0020396, 0.0019326
7: -0.0212852, -0.0133860, -0.0215259, -0.0136250, -0.0061679, 0.0066993
8: 0.9628063, 0.9854385, 0.9621167, 0.9847537, -0.0197419, 0.0216809
9: 0.0019328, 0.0085844, 0.0021340, 0.0087871, -0.0058236, 0.0053458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0145245
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0145251
time: 1.01 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004641, 0.0007781, -0.0004830, 0.0006363, -0.0009479, 0.0011033
1: -0.0008525, 0.0025141, -0.0009410, 0.0022967, -0.0029624, 0.0033185
2: 0.0125748, 0.0176167, 0.0129004, 0.0177492, -0.0048070, 0.0042758
3: -0.0011711, 0.0026201, -0.0009264, 0.0027198, -0.0035446, 0.0031499
4: -0.0054599, -0.0019629, -0.0052341, -0.0018709, -0.0035890, 0.0032713
5: 0.0067692, 0.0105535, 0.0070135, 0.0106530, -0.0035316, 0.0031383
6: 0.0080655, 0.0103784, 0.0083951, 0.0102862, -0.0022207, 0.0019833
7: -0.0213099, -0.0130946, -0.0215259, -0.0136250, -0.0062483, 0.0070222
8: 0.9627355, 0.9862733, 0.9621167, 0.9847537, -0.0201204, 0.0226060
9: 0.0016874, 0.0086052, 0.0021340, 0.0087871, -0.0060955, 0.0054243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0146862
time: 0.96 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144342, upper bound: 0.0146894
time: 0.96 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005023, 0.0006591, -0.0004595, 0.0006532, -0.0009987, 0.0009704
1: -0.0010312, 0.0023318, -0.0008307, 0.0023227, -0.0031849, 0.0029219
2: 0.0128479, 0.0178844, 0.0128614, 0.0175841, -0.0042276, 0.0046224
3: -0.0009658, 0.0028214, -0.0009556, 0.0025956, -0.0031183, 0.0034116
4: -0.0052705, -0.0017772, -0.0052611, -0.0019855, -0.0032850, 0.0034839
5: 0.0069741, 0.0107544, 0.0069843, 0.0105290, -0.0031071, 0.0033993
6: 0.0083420, 0.0103011, 0.0083557, 0.0102972, -0.0019552, 0.0019453
7: -0.0217461, -0.0135396, -0.0212568, -0.0135616, -0.0067745, 0.0062691
8: 0.9614856, 0.9849985, 0.9628878, 0.9849353, -0.0217287, 0.0198829
9: 0.0020621, 0.0089725, 0.0020806, 0.0085605, -0.0054048, 0.0058791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0140437
time: 0.85 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0142317
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005018, 0.0006515, -0.0004798, 0.0006555, -0.0010060, 0.0009836
1: -0.0010290, 0.0023201, -0.0009258, 0.0023262, -0.0032049, 0.0029775
2: 0.0128654, 0.0178811, 0.0128563, 0.0177264, -0.0043223, 0.0046551
3: -0.0009527, 0.0028189, -0.0009595, 0.0027026, -0.0031948, 0.0034372
4: -0.0052584, -0.0017795, -0.0052647, -0.0018867, -0.0033716, 0.0034852
5: 0.0069872, 0.0107519, 0.0069804, 0.0106358, -0.0031840, 0.0034250
6: 0.0083597, 0.0102961, 0.0083505, 0.0102987, -0.0019390, 0.0019457
7: -0.0217407, -0.0135680, -0.0214887, -0.0135531, -0.0068414, 0.0064821
8: 0.9615012, 0.9849170, 0.9622233, 0.9849596, -0.0218784, 0.0203135
9: 0.0020860, 0.0089680, 0.0020735, 0.0087558, -0.0055785, 0.0059327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0138452
time: 0.85 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0140612
time: 0.84 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005023, 0.0006591, -0.0004492, 0.0006869, -0.0010307, 0.0009638
1: -0.0010312, 0.0023318, -0.0007826, 0.0023743, -0.0032365, 0.0029160
2: 0.0128479, 0.0178844, 0.0127842, 0.0175120, -0.0042061, 0.0046997
3: -0.0009658, 0.0028214, -0.0010138, 0.0025414, -0.0030989, 0.0034697
4: -0.0052705, -0.0017772, -0.0053147, -0.0020354, -0.0032350, 0.0035375
5: 0.0069741, 0.0107544, 0.0069263, 0.0104749, -0.0030876, 0.0034573
6: 0.0083420, 0.0103011, 0.0082774, 0.0103191, -0.0019771, 0.0020236
7: -0.0217461, -0.0135396, -0.0211394, -0.0134356, -0.0069004, 0.0062037
8: 0.9614856, 0.9849985, 0.9632240, 0.9852962, -0.0220895, 0.0197934
9: 0.0020621, 0.0089725, 0.0019746, 0.0084616, -0.0053598, 0.0059851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139452, upper bound: 0.0142643
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139452, upper bound: 0.0142643
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005018, 0.0006515, -0.0004697, 0.0006869, -0.0010370, 0.0009771
1: -0.0010290, 0.0023201, -0.0008785, 0.0023743, -0.0032569, 0.0029663
2: 0.0128654, 0.0178811, 0.0127842, 0.0176557, -0.0042988, 0.0047328
3: -0.0009527, 0.0028189, -0.0010137, 0.0026494, -0.0031742, 0.0034957
4: -0.0052584, -0.0017795, -0.0053147, -0.0019358, -0.0033226, 0.0035352
5: 0.0069872, 0.0107519, 0.0069263, 0.0105828, -0.0031634, 0.0034834
6: 0.0083597, 0.0102961, 0.0082775, 0.0103191, -0.0019594, 0.0020186
7: -0.0217407, -0.0135680, -0.0213734, -0.0134358, -0.0069681, 0.0064126
8: 0.9615012, 0.9849170, 0.9625535, 0.9852958, -0.0222415, 0.0202105
9: 0.0020860, 0.0089680, 0.0019747, 0.0086587, -0.0055283, 0.0060394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139671, upper bound: 0.0141114
time: 1.14 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139671, upper bound: 0.0141114
time: 0.98 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0004916, 0.0005596, -0.0004804, 0.0007054, -0.0010309, 0.0008682
1: -0.0009813, 0.0021792, -0.0009287, 0.0024027, -0.0029888, 0.0029219
2: 0.0130764, 0.0178095, 0.0127416, 0.0177307, -0.0041961, 0.0043629
3: -0.0007940, 0.0027651, -0.0010457, 0.0027059, -0.0030770, 0.0032356
4: -0.0051120, -0.0018291, -0.0053442, -0.0018837, -0.0032283, 0.0034338
5: 0.0071456, 0.0106983, 0.0068943, 0.0106391, -0.0030642, 0.0032256
6: 0.0085734, 0.0102363, 0.0082344, 0.0103312, -0.0017577, 0.0020020
7: -0.0216241, -0.0139119, -0.0214958, -0.0133663, -0.0066222, 0.0059650
8: 0.9618351, 0.9839317, 0.9622030, 0.9854949, -0.0204794, 0.0197707
9: 0.0023756, 0.0088698, 0.0019162, 0.0087617, -0.0052114, 0.0056833

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
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146138, upper bound: 0.0145233
time: 0.91 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146138, upper bound: 0.0145233
time: 1.05 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004620, 0.0007002, -0.0004804, 0.0007054, -0.0010163, 0.0010219
1: -0.0008424, 0.0023947, -0.0009287, 0.0024027, -0.0030340, 0.0032168
2: 0.0127537, 0.0176015, 0.0127416, 0.0177307, -0.0046376, 0.0043949
3: -0.0010367, 0.0026087, -0.0010457, 0.0027059, -0.0034091, 0.0032431
4: -0.0053359, -0.0019734, -0.0053442, -0.0018837, -0.0034521, 0.0033709
5: 0.0069034, 0.0105421, 0.0068943, 0.0106391, -0.0033957, 0.0032315
6: 0.0082466, 0.0103278, 0.0082344, 0.0103312, -0.0020846, 0.0020934
7: -0.0212852, -0.0133860, -0.0214958, -0.0133663, -0.0064886, 0.0066844
8: 0.9628063, 0.9854385, 0.9622030, 0.9854949, -0.0206606, 0.0218320
9: 0.0019328, 0.0085844, 0.0019162, 0.0087617, -0.0058172, 0.0056158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146138, upper bound: 0.0145233
time: 0.88 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146138, upper bound: 0.0145233
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004907, 0.0006316, -0.0004804, 0.0007054, -0.0010169, 0.0009383
1: -0.0009770, 0.0022896, -0.0009287, 0.0024027, -0.0029792, 0.0029140
2: 0.0129111, 0.0178031, 0.0127416, 0.0177307, -0.0041985, 0.0043223
3: -0.0009183, 0.0027603, -0.0010457, 0.0027059, -0.0030910, 0.0031942
4: -0.0052267, -0.0018335, -0.0053442, -0.0018837, -0.0033430, 0.0034953
5: 0.0070215, 0.0106934, 0.0068943, 0.0106391, -0.0030793, 0.0031834
6: 0.0084060, 0.0102832, 0.0082344, 0.0103312, -0.0019252, 0.0020488
7: -0.0216137, -0.0136425, -0.0214958, -0.0133663, -0.0064627, 0.0061224
8: 0.9618650, 0.9847037, 0.9622030, 0.9854949, -0.0203170, 0.0197605
9: 0.0021487, 0.0088610, 0.0019162, 0.0087617, -0.0053118, 0.0055604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144595, upper bound: 0.0146862
time: 1.00 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144595, upper bound: 0.0146894
time: 1.14 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004641, 0.0007781, -0.0004804, 0.0007054, -0.0010035, 0.0010883
1: -0.0008525, 0.0025141, -0.0009287, 0.0024027, -0.0030378, 0.0033251
2: 0.0125748, 0.0176167, 0.0127416, 0.0177307, -0.0047890, 0.0043697
3: -0.0011711, 0.0026201, -0.0010457, 0.0027059, -0.0035184, 0.0032148
4: -0.0054599, -0.0019629, -0.0053442, -0.0018837, -0.0035762, 0.0033814
5: 0.0067692, 0.0105535, 0.0068943, 0.0106391, -0.0035043, 0.0032024
6: 0.0080655, 0.0103784, 0.0082344, 0.0103312, -0.0022657, 0.0021440
7: -0.0213099, -0.0130946, -0.0214958, -0.0133663, -0.0063485, 0.0068563
8: 0.9627355, 0.9862733, 0.9622030, 0.9854949, -0.0205766, 0.0225512
9: 0.0016874, 0.0086052, 0.0019162, 0.0087617, -0.0059824, 0.0055163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144595, upper bound: 0.0146862
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144595, upper bound: 0.0146894
time: 1.15 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005011, 0.0005614, -0.0004641, 0.0007781, -0.0011187, 0.0008818
1: -0.0010258, 0.0021820, -0.0008525, 0.0025141, -0.0033020, 0.0028803
2: 0.0130721, 0.0178762, 0.0125748, 0.0176167, -0.0041350, 0.0048110
3: -0.0007972, 0.0028153, -0.0011711, 0.0026201, -0.0030383, 0.0035589
4: -0.0051150, -0.0017828, -0.0054599, -0.0019629, -0.0031521, 0.0036771
5: 0.0071424, 0.0107483, 0.0067692, 0.0105535, -0.0030265, 0.0035471
6: 0.0085691, 0.0102376, 0.0080655, 0.0103784, -0.0018093, 0.0021721
7: -0.0217328, -0.0139049, -0.0213099, -0.0130946, -0.0071752, 0.0060229
8: 0.9615238, 0.9839517, 0.9627355, 0.9862733, -0.0225980, 0.0194817
9: 0.0023697, 0.0089613, 0.0016874, 0.0086052, -0.0052172, 0.0061924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0140437
time: 1.09 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0138452
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0005011, 0.0006335, -0.0004641, 0.0007781, -0.0011058, 0.0009396
1: -0.0010255, 0.0022925, -0.0008525, 0.0025141, -0.0033040, 0.0029648
2: 0.0129067, 0.0178757, 0.0125748, 0.0176167, -0.0042410, 0.0047927
3: -0.0009216, 0.0028149, -0.0011711, 0.0026201, -0.0031077, 0.0035353
4: -0.0052297, -0.0017832, -0.0054599, -0.0019629, -0.0032669, 0.0036767
5: 0.0070183, 0.0107479, 0.0067692, 0.0105535, -0.0030948, 0.0035225
6: 0.0084016, 0.0102844, 0.0080655, 0.0103784, -0.0019768, 0.0022189
7: -0.0217320, -0.0136353, -0.0213099, -0.0130946, -0.0070299, 0.0061106
8: 0.9615262, 0.9847240, 0.9627355, 0.9862733, -0.0225324, 0.0199978
9: 0.0021427, 0.0089606, 0.0016874, 0.0086052, -0.0052948, 0.0060931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138349, upper bound: 0.0142353
time: 1.37 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138452, upper bound: 0.0140773
time: 1.08 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005011, 0.0005614, -0.0004502, 0.0008016, -0.0011412, 0.0008699
1: -0.0010258, 0.0021820, -0.0007871, 0.0025501, -0.0033404, 0.0028557
2: 0.0130721, 0.0178762, 0.0125209, 0.0175188, -0.0040866, 0.0048685
3: -0.0007972, 0.0028153, -0.0012117, 0.0025465, -0.0029985, 0.0036021
4: -0.0051150, -0.0017828, -0.0054973, -0.0020307, -0.0030842, 0.0037145
5: 0.0071424, 0.0107483, 0.0067286, 0.0104800, -0.0029866, 0.0035902
6: 0.0085691, 0.0102376, 0.0080108, 0.0103937, -0.0018246, 0.0022267
7: -0.0217328, -0.0139049, -0.0211504, -0.0130066, -0.0072689, 0.0058954
8: 0.9615238, 0.9839517, 0.9631923, 0.9865254, -0.0228665, 0.0192632
9: 0.0023697, 0.0089613, 0.0016133, 0.0084709, -0.0051243, 0.0062713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139452, upper bound: 0.0140570
time: 1.11 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139671, upper bound: 0.0138816
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0005011, 0.0006335, -0.0004502, 0.0008016, -0.0011287, 0.0009312
1: -0.0010255, 0.0022925, -0.0007871, 0.0025501, -0.0033459, 0.0029584
2: 0.0129067, 0.0178757, 0.0125209, 0.0175188, -0.0042176, 0.0048554
3: -0.0009216, 0.0028149, -0.0012117, 0.0025465, -0.0030840, 0.0035824
4: -0.0052297, -0.0017832, -0.0054973, -0.0020307, -0.0031990, 0.0037142
5: 0.0070183, 0.0107479, 0.0067286, 0.0104800, -0.0030708, 0.0035695
6: 0.0084016, 0.0102844, 0.0080108, 0.0103937, -0.0019921, 0.0022736
7: -0.0217320, -0.0136353, -0.0211504, -0.0130066, -0.0071320, 0.0060294
8: 0.9615262, 0.9847240, 0.9631923, 0.9865254, -0.0228251, 0.0199013
9: 0.0021427, 0.0089606, 0.0016133, 0.0084709, -0.0052432, 0.0061791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139452, upper bound: 0.0142661
time: 0.97 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139671, upper bound: 0.0141232
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004804, 0.0006905, -0.0004916, 0.0005596, -0.0008668, 0.0010123
1: -0.0009287, 0.0023799, -0.0009813, 0.0021792, -0.0028827, 0.0029451
2: 0.0127759, 0.0177308, 0.0130764, 0.0178095, -0.0042974, 0.0041456
3: -0.0010200, 0.0027059, -0.0007940, 0.0027651, -0.0031864, 0.0030433
4: -0.0053205, -0.0018837, -0.0051120, -0.0018291, -0.0033884, 0.0032283
5: 0.0069200, 0.0106392, 0.0071456, 0.0106983, -0.0031764, 0.0030309
6: 0.0082691, 0.0103215, 0.0085734, 0.0102363, -0.0019673, 0.0017480
7: -0.0214959, -0.0134221, -0.0216241, -0.0139119, -0.0059376, 0.0065155
8: 0.9622025, 0.9853348, 0.9618351, 0.9839317, -0.0195255, 0.0201738
9: 0.0019632, 0.0087618, 0.0023756, 0.0088698, -0.0055934, 0.0051751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0146264
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008248, -0.0004916, 0.0005596, -0.0008529, 0.0011484
1: -0.0007930, 0.0025856, -0.0009813, 0.0021792, -0.0029165, 0.0033041
2: 0.0124677, 0.0175276, 0.0130764, 0.0178095, -0.0048088, 0.0041601
3: -0.0012517, 0.0025531, -0.0007940, 0.0027651, -0.0035562, 0.0030418
4: -0.0055342, -0.0020246, -0.0051120, -0.0018291, -0.0037051, 0.0030874
5: 0.0066887, 0.0104866, 0.0071456, 0.0106983, -0.0035441, 0.0030284
6: 0.0079570, 0.0104088, 0.0085734, 0.0102363, -0.0022794, 0.0018353
7: -0.0211648, -0.0129200, -0.0216241, -0.0139119, -0.0058127, 0.0071615
8: 0.9631513, 0.9867736, 0.9618351, 0.9839317, -0.0196322, 0.0225905
9: 0.0015404, 0.0084830, 0.0023756, 0.0088698, -0.0061733, 0.0051149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0146264
time: 0.82 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147476
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004804, 0.0006905, -0.0004819, 0.0005993, -0.0009196, 0.0010085
1: -0.0009287, 0.0023799, -0.0009355, 0.0022401, -0.0029217, 0.0029903
2: 0.0127759, 0.0177308, 0.0129853, 0.0177411, -0.0043449, 0.0042151
3: -0.0010200, 0.0027059, -0.0008625, 0.0027136, -0.0032133, 0.0031064
4: -0.0053205, -0.0018837, -0.0051752, -0.0018766, -0.0034439, 0.0032916
5: 0.0069200, 0.0106392, 0.0070772, 0.0106468, -0.0032025, 0.0030950
6: 0.0082691, 0.0103215, 0.0084811, 0.0102622, -0.0019931, 0.0018404
7: -0.0214959, -0.0134221, -0.0215126, -0.0137633, -0.0061977, 0.0064937
8: 0.9622025, 0.9853348, 0.9621548, 0.9843574, -0.0198375, 0.0204128
9: 0.0019632, 0.0087618, 0.0022505, 0.0087759, -0.0055939, 0.0053640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0147486
time: 1.06 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0148799
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004514, 0.0008248, -0.0004819, 0.0005993, -0.0009081, 0.0011458
1: -0.0007930, 0.0025856, -0.0009355, 0.0022401, -0.0029719, 0.0033562
2: 0.0124677, 0.0175276, 0.0129853, 0.0177411, -0.0048710, 0.0042611
3: -0.0012517, 0.0025531, -0.0008625, 0.0027136, -0.0035946, 0.0031253
4: -0.0055342, -0.0020246, -0.0051752, -0.0018766, -0.0036576, 0.0031506
5: 0.0066887, 0.0104866, 0.0070772, 0.0106468, -0.0035818, 0.0031126
6: 0.0079570, 0.0104088, 0.0084811, 0.0102622, -0.0023052, 0.0019276
7: -0.0211648, -0.0129200, -0.0215126, -0.0137633, -0.0060957, 0.0071669
8: 0.9631513, 0.9867736, 0.9621548, 0.9843574, -0.0200814, 0.0228977
9: 0.0015404, 0.0084830, 0.0022505, 0.0087759, -0.0061970, 0.0053231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0147486
time: 1.00 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0148799
time: 0.85 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0004897, 0.0006926, -0.0004595, 0.0006532, -0.0009883, 0.0010037
1: -0.0009724, 0.0023831, -0.0008307, 0.0023227, -0.0031638, 0.0029812
2: 0.0127710, 0.0177962, 0.0128614, 0.0175841, -0.0043164, 0.0045734
3: -0.0010236, 0.0027551, -0.0009556, 0.0025956, -0.0031851, 0.0033689
4: -0.0053238, -0.0018383, -0.0052611, -0.0019855, -0.0033384, 0.0034228
5: 0.0069164, 0.0106882, 0.0069843, 0.0105290, -0.0031738, 0.0033562
6: 0.0082641, 0.0103228, 0.0083557, 0.0102972, -0.0020331, 0.0019671
7: -0.0216024, -0.0134142, -0.0212568, -0.0135616, -0.0066562, 0.0064139
8: 0.9618973, 0.9853576, 0.9628878, 0.9849353, -0.0215170, 0.0202976
9: 0.0019566, 0.0088515, 0.0020806, 0.0085605, -0.0055267, 0.0057805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B2_B1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0141657
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0143094
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0004892, 0.0006851, -0.0004798, 0.0006555, -0.0009956, 0.0010170
1: -0.0009700, 0.0023716, -0.0009258, 0.0023262, -0.0031830, 0.0030368
2: 0.0127883, 0.0177927, 0.0128563, 0.0177264, -0.0044111, 0.0046049
3: -0.0010106, 0.0027525, -0.0009595, 0.0027026, -0.0032616, 0.0033940
4: -0.0053119, -0.0018407, -0.0052647, -0.0018867, -0.0034251, 0.0034240
5: 0.0069294, 0.0106856, 0.0069804, 0.0106358, -0.0032506, 0.0033814
6: 0.0082816, 0.0103179, 0.0083505, 0.0102987, -0.0020171, 0.0019675
7: -0.0215968, -0.0134424, -0.0214887, -0.0135531, -0.0067226, 0.0066268
8: 0.9619135, 0.9852769, 0.9622233, 0.9849596, -0.0216611, 0.0207281
9: 0.0019803, 0.0088468, 0.0020735, 0.0087558, -0.0057003, 0.0058339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0139671
time: 0.83 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0141282
time: 0.83 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0004897, 0.0006926, -0.0004492, 0.0006869, -0.0010247, 0.0010001
1: -0.0009724, 0.0023831, -0.0007826, 0.0023743, -0.0032731, 0.0030274
2: 0.0127710, 0.0177962, 0.0127842, 0.0175120, -0.0043671, 0.0047398
3: -0.0010236, 0.0027551, -0.0010138, 0.0025414, -0.0032146, 0.0034931
4: -0.0053238, -0.0018383, -0.0053147, -0.0020354, -0.0032884, 0.0034764
5: 0.0069164, 0.0106882, 0.0069263, 0.0104749, -0.0032025, 0.0034800
6: 0.0082641, 0.0103228, 0.0082774, 0.0103191, -0.0020550, 0.0020454
7: -0.0216024, -0.0134142, -0.0211394, -0.0134356, -0.0068790, 0.0063949
8: 0.9618973, 0.9853576, 0.9632240, 0.9852962, -0.0222917, 0.0205557
9: 0.0019566, 0.0088515, 0.0019746, 0.0084616, -0.0055290, 0.0059887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141763, upper bound: 0.0143569
time: 0.87 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141763, upper bound: 0.0145156
time: 0.89 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0004892, 0.0006851, -0.0004697, 0.0006869, -0.0010307, 0.0010127
1: -0.0009700, 0.0023716, -0.0008785, 0.0023743, -0.0032926, 0.0030736
2: 0.0127883, 0.0177927, 0.0127842, 0.0176557, -0.0044513, 0.0047717
3: -0.0010106, 0.0027525, -0.0010137, 0.0026494, -0.0032846, 0.0035182
4: -0.0053119, -0.0018407, -0.0053147, -0.0019358, -0.0033760, 0.0034739
5: 0.0069294, 0.0106856, 0.0069263, 0.0105828, -0.0032728, 0.0035051
6: 0.0082816, 0.0103179, 0.0082775, 0.0103191, -0.0020375, 0.0020404
7: -0.0215968, -0.0134424, -0.0213734, -0.0134358, -0.0069454, 0.0065963
8: 0.9619135, 0.9852769, 0.9625535, 0.9852958, -0.0224384, 0.0209299
9: 0.0019803, 0.0088468, 0.0019747, 0.0086587, -0.0056902, 0.0060421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A1_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142080, upper bound: 0.0142187
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142080, upper bound: 0.0143839
time: 0.90 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0005009, 0.0006040, -0.0004907, 0.0006316, -0.0009719, 0.0009301
1: -0.0010247, 0.0022474, -0.0009770, 0.0022896, -0.0029510, 0.0028621
2: 0.0129743, 0.0178746, 0.0129111, 0.0178031, -0.0041618, 0.0043056
3: -0.0008708, 0.0028141, -0.0009183, 0.0027603, -0.0030795, 0.0031915
4: -0.0051828, -0.0017839, -0.0052267, -0.0018335, -0.0033371, 0.0034131
5: 0.0070690, 0.0107471, 0.0070215, 0.0106934, -0.0030693, 0.0031814
6: 0.0084700, 0.0102653, 0.0084060, 0.0102832, -0.0018131, 0.0018593
7: -0.0217302, -0.0137455, -0.0216137, -0.0136425, -0.0064819, 0.0062398
8: 0.9615313, 0.9844084, 0.9618650, 0.9847037, -0.0202109, 0.0195520
9: 0.0022355, 0.0089591, 0.0021487, 0.0088610, -0.0053766, 0.0055839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B2_B1_B1_A1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0146264
time: 1.07 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0146264
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004985, 0.0006698, -0.0004907, 0.0006316, -0.0009557, 0.0009821
1: -0.0010132, 0.0023482, -0.0009770, 0.0022896, -0.0029350, 0.0029247
2: 0.0128233, 0.0178574, 0.0129111, 0.0178031, -0.0042408, 0.0042537
3: -0.0009843, 0.0028011, -0.0009183, 0.0027603, -0.0031329, 0.0031435
4: -0.0052875, -0.0017959, -0.0052267, -0.0018335, -0.0034388, 0.0034308
5: 0.0069557, 0.0107342, 0.0070215, 0.0106934, -0.0031222, 0.0031327
6: 0.0083171, 0.0103080, 0.0084060, 0.0102832, -0.0019660, 0.0019021
7: -0.0217021, -0.0134995, -0.0216137, -0.0136425, -0.0063089, 0.0063299
8: 0.9616117, 0.9851132, 0.9618650, 0.9847037, -0.0199954, 0.0199365
9: 0.0020284, 0.0089355, 0.0021487, 0.0088610, -0.0054486, 0.0054553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147477
time: 1.08 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144379, upper bound: 0.0147477
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0005009, 0.0006040, -0.0004792, 0.0006651, -0.0010091, 0.0009233
1: -0.0010247, 0.0022474, -0.0009232, 0.0023410, -0.0030698, 0.0028977
2: 0.0129743, 0.0178746, 0.0128341, 0.0177226, -0.0041944, 0.0044718
3: -0.0008708, 0.0028141, -0.0009762, 0.0026997, -0.0030950, 0.0033102
4: -0.0051828, -0.0017839, -0.0052801, -0.0018894, -0.0032934, 0.0034961
5: 0.0070690, 0.0107471, 0.0069638, 0.0106330, -0.0030841, 0.0032992
6: 0.0084700, 0.0102653, 0.0083280, 0.0103050, -0.0018349, 0.0019372
7: -0.0217302, -0.0137455, -0.0214825, -0.0135171, -0.0066895, 0.0061980
8: 0.9615313, 0.9844084, 0.9622412, 0.9850628, -0.0209994, 0.0197224
9: 0.0022355, 0.0089591, 0.0020432, 0.0087505, -0.0053587, 0.0057644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0147475
time: 1.20 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0147475
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004985, 0.0006698, -0.0004792, 0.0006651, -0.0009938, 0.0009773
1: -0.0010132, 0.0023482, -0.0009232, 0.0023410, -0.0030486, 0.0029616
2: 0.0128233, 0.0178574, 0.0128341, 0.0177226, -0.0042748, 0.0044161
3: -0.0009843, 0.0028011, -0.0009762, 0.0026997, -0.0031496, 0.0032596
4: -0.0052875, -0.0017959, -0.0052801, -0.0018894, -0.0033982, 0.0034842
5: 0.0069557, 0.0107342, 0.0069638, 0.0106330, -0.0031380, 0.0032480
6: 0.0083171, 0.0103080, 0.0083280, 0.0103050, -0.0019878, 0.0019800
7: -0.0217021, -0.0134995, -0.0214825, -0.0135171, -0.0065179, 0.0062835
8: 0.9616117, 0.9851132, 0.9622412, 0.9850628, -0.0207661, 0.0201171
9: 0.0020284, 0.0089355, 0.0020432, 0.0087505, -0.0054299, 0.0056304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_B1_A1_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0148799
time: 1.03 seconds

## Relational analysis of IS_B1_A1_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146632, upper bound: 0.0148799
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004910, 0.0006014, -0.0004641, 0.0007781, -0.0011137, 0.0009220
1: -0.0009785, 0.0022433, -0.0008525, 0.0025141, -0.0033035, 0.0029481
2: 0.0129804, 0.0178055, 0.0125748, 0.0176167, -0.0042366, 0.0048043
3: -0.0008662, 0.0027621, -0.0011711, 0.0026201, -0.0031147, 0.0035506
4: -0.0051786, -0.0018319, -0.0054599, -0.0019629, -0.0032157, 0.0036280
5: 0.0070736, 0.0106952, 0.0067692, 0.0105535, -0.0031028, 0.0035383
6: 0.0084762, 0.0102635, 0.0080655, 0.0103784, -0.0019022, 0.0021980
7: -0.0216175, -0.0137554, -0.0213099, -0.0130946, -0.0071157, 0.0061884
8: 0.9618542, 0.9843799, 0.9627355, 0.9862733, -0.0225733, 0.0199559
9: 0.0022439, 0.0088643, 0.0016874, 0.0086052, -0.0053565, 0.0061528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0141657
time: 1.01 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0139671
time: 1.00 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004885, 0.0006672, -0.0004641, 0.0007781, -0.0010992, 0.0009726
1: -0.0009667, 0.0023442, -0.0008525, 0.0025141, -0.0033029, 0.0030275
2: 0.0128294, 0.0177878, 0.0125748, 0.0176167, -0.0043349, 0.0047768
3: -0.0009798, 0.0027487, -0.0011711, 0.0026201, -0.0031783, 0.0035197
4: -0.0052834, -0.0018442, -0.0054599, -0.0019629, -0.0033205, 0.0036157
5: 0.0069602, 0.0106819, 0.0067692, 0.0105535, -0.0031653, 0.0035066
6: 0.0083232, 0.0103063, 0.0080655, 0.0103784, -0.0020552, 0.0022408
7: -0.0215887, -0.0135093, -0.0213099, -0.0130946, -0.0069592, 0.0062636
8: 0.9619369, 0.9850852, 0.9627355, 0.9862733, -0.0224715, 0.0204363
9: 0.0020366, 0.0088399, 0.0016874, 0.0086052, -0.0054237, 0.0060397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138657, upper bound: 0.0143137
time: 1.04 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0138816, upper bound: 0.0141461
time: 1.06 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0004910, 0.0006014, -0.0004502, 0.0008016, -0.0011389, 0.0009155
1: -0.0009785, 0.0022433, -0.0007871, 0.0025501, -0.0033942, 0.0029794
2: 0.0129804, 0.0178055, 0.0125209, 0.0175188, -0.0042695, 0.0049359
3: -0.0008662, 0.0027621, -0.0012117, 0.0025465, -0.0031303, 0.0036463
4: -0.0051786, -0.0018319, -0.0054973, -0.0020307, -0.0031479, 0.0036654
5: 0.0070736, 0.0106952, 0.0067286, 0.0104800, -0.0031174, 0.0036336
6: 0.0084762, 0.0102635, 0.0080108, 0.0103937, -0.0019175, 0.0022527
7: -0.0216175, -0.0137554, -0.0211504, -0.0130066, -0.0072809, 0.0061486
8: 0.9618542, 0.9843799, 0.9631923, 0.9865254, -0.0231941, 0.0201245
9: 0.0022439, 0.0088643, 0.0016133, 0.0084709, -0.0053394, 0.0063028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141763, upper bound: 0.0143569
time: 1.14 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142080, upper bound: 0.0142187
time: 1.10 seconds

## BFS IS instance: IS_B1_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004885, 0.0006672, -0.0004502, 0.0008016, -0.0011254, 0.0009679
1: -0.0009667, 0.0023442, -0.0007871, 0.0025501, -0.0033875, 0.0030571
2: 0.0128294, 0.0177878, 0.0125209, 0.0175188, -0.0043667, 0.0049053
3: -0.0009798, 0.0027487, -0.0012117, 0.0025465, -0.0031945, 0.0036145
4: -0.0052834, -0.0018442, -0.0054973, -0.0020307, -0.0032526, 0.0036532
5: 0.0069602, 0.0106819, 0.0067286, 0.0104800, -0.0031806, 0.0036009
6: 0.0083232, 0.0103063, 0.0080108, 0.0103937, -0.0020705, 0.0022955
7: -0.0215887, -0.0135093, -0.0211504, -0.0130066, -0.0071249, 0.0062179
8: 0.9619369, 0.9850852, 0.9631923, 0.9865254, -0.0230692, 0.0205994
9: 0.0020366, 0.0088399, 0.0016133, 0.0084709, -0.0054038, 0.0061916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141763, upper bound: 0.0145156
time: 1.19 seconds

## Relational analysis of IS_B1_A1_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142080, upper bound: 0.0143839
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005087, 0.0007719, -0.0004916, 0.0005596, -0.0009109, 0.0011111
1: -0.0010613, 0.0025046, -0.0009813, 0.0021792, -0.0030624, 0.0031600
2: 0.0125891, 0.0179294, 0.0130764, 0.0178095, -0.0046193, 0.0044174
3: -0.0011604, 0.0028553, -0.0007940, 0.0027651, -0.0034284, 0.0032534
4: -0.0054500, -0.0017459, -0.0051120, -0.0018291, -0.0036117, 0.0033661
5: 0.0067799, 0.0107882, 0.0071456, 0.0106983, -0.0034180, 0.0032411
6: 0.0080800, 0.0103744, 0.0085734, 0.0102363, -0.0021564, 0.0018009
7: -0.0218195, -0.0131179, -0.0216241, -0.0139119, -0.0064377, 0.0070400
8: 0.9612754, 0.9862067, 0.9618351, 0.9839317, -0.0207860, 0.0216764
9: 0.0017070, 0.0090343, 0.0023756, 0.0088698, -0.0060350, 0.0055894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152471, upper bound: 0.0148962
time: 1.08 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152471, upper bound: 0.0150484
time: 1.11 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005087, 0.0007719, -0.0004819, 0.0005993, -0.0009591, 0.0011035
1: -0.0010613, 0.0025046, -0.0009355, 0.0022401, -0.0030655, 0.0031514
2: 0.0125891, 0.0179294, 0.0129853, 0.0177411, -0.0046002, 0.0044513
3: -0.0011604, 0.0028553, -0.0008625, 0.0027136, -0.0034103, 0.0032914
4: -0.0054500, -0.0017459, -0.0051752, -0.0018766, -0.0035734, 0.0034293
5: 0.0067799, 0.0107882, 0.0070772, 0.0106468, -0.0033997, 0.0032804
6: 0.0080800, 0.0103744, 0.0084811, 0.0102622, -0.0021822, 0.0018932
7: -0.0218195, -0.0131179, -0.0215126, -0.0137633, -0.0066293, 0.0069648
8: 0.9612754, 0.9862067, 0.9621548, 0.9843574, -0.0209144, 0.0215941
9: 0.0017070, 0.0090343, 0.0022505, 0.0087759, -0.0059849, 0.0057272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152471, upper bound: 0.0148962
time: 1.19 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0152471, upper bound: 0.0150484
time: 1.27 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004787, 0.0008152, -0.0004830, 0.0006363, -0.0009719, 0.0011515
1: -0.0009209, 0.0025710, -0.0009410, 0.0022967, -0.0030654, 0.0034285
2: 0.0124896, 0.0177192, 0.0129004, 0.0177492, -0.0049717, 0.0044405
3: -0.0012352, 0.0026972, -0.0009264, 0.0027198, -0.0036684, 0.0032783
4: -0.0055190, -0.0018918, -0.0052341, -0.0018709, -0.0036481, 0.0033423
5: 0.0067052, 0.0106304, 0.0070135, 0.0106530, -0.0036552, 0.0032667
6: 0.0079792, 0.0104025, 0.0083951, 0.0102862, -0.0023070, 0.0020074
7: -0.0214769, -0.0129557, -0.0215259, -0.0136250, -0.0065217, 0.0072906
8: 0.9622570, 0.9866712, 0.9621167, 0.9847537, -0.0208809, 0.0233749
9: 0.0015705, 0.0087458, 0.0021340, 0.0087871, -0.0063215, 0.0056564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B1_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145377, upper bound: 0.0140785
time: 0.92 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143245, upper bound: 0.0141060
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004793, 0.0008813, -0.0004830, 0.0006363, -0.0009755, 0.0012198
1: -0.0009236, 0.0026723, -0.0009410, 0.0022967, -0.0031073, 0.0035393
2: 0.0123380, 0.0177232, 0.0129004, 0.0177492, -0.0051376, 0.0044881
3: -0.0013492, 0.0027002, -0.0009264, 0.0027198, -0.0037932, 0.0033087
4: -0.0056242, -0.0018890, -0.0052341, -0.0018709, -0.0037533, 0.0033451
5: 0.0065914, 0.0106334, 0.0070135, 0.0106530, -0.0037798, 0.0032969
6: 0.0078256, 0.0104455, 0.0083951, 0.0102862, -0.0024606, 0.0020504
7: -0.0214834, -0.0127087, -0.0215259, -0.0136250, -0.0065623, 0.0075609
8: 0.9622383, 0.9873791, 0.9621167, 0.9847537, -0.0211155, 0.0241494
9: 0.0013624, 0.0087514, 0.0021340, 0.0087871, -0.0065491, 0.0056946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B1_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0145377, upper bound: 0.0142818
time: 0.89 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0143245, upper bound: 0.0143239
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004595, 0.0006532, -0.0010307, 0.0011024
1: -0.0011087, 0.0025076, -0.0008307, 0.0023227, -0.0033466, 0.0031960
2: 0.0125846, 0.0180004, 0.0128614, 0.0175841, -0.0046381, 0.0048499
3: -0.0011638, 0.0029086, -0.0009556, 0.0025956, -0.0034270, 0.0035783
4: -0.0054531, -0.0016967, -0.0052611, -0.0019855, -0.0034676, 0.0035644
5: 0.0067765, 0.0108415, 0.0069843, 0.0105290, -0.0034152, 0.0035653
6: 0.0080754, 0.0103756, 0.0083557, 0.0102972, -0.0022218, 0.0020199
7: -0.0219351, -0.0131106, -0.0212568, -0.0135616, -0.0071394, 0.0069380
8: 0.9609442, 0.9862276, 0.9628878, 0.9849353, -0.0228027, 0.0217993
9: 0.0017009, 0.0091317, 0.0020806, 0.0085605, -0.0059681, 0.0061753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0139908
time: 0.99 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0139994, upper bound: 0.0142001
time: 1.01 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0005184, 0.0007667, -0.0004798, 0.0006555, -0.0010380, 0.0011158
1: -0.0011065, 0.0024967, -0.0009258, 0.0023262, -0.0033664, 0.0032507
2: 0.0126010, 0.0179970, 0.0128563, 0.0177264, -0.0047315, 0.0048830
3: -0.0011515, 0.0029061, -0.0009595, 0.0027026, -0.0035025, 0.0036042
4: -0.0054418, -0.0016990, -0.0052647, -0.0018867, -0.0035550, 0.0035657
5: 0.0067888, 0.0108390, 0.0069804, 0.0106358, -0.0034911, 0.0035914
6: 0.0080919, 0.0103710, 0.0083505, 0.0102987, -0.0022068, 0.0020205
7: -0.0219297, -0.0131372, -0.0214887, -0.0135531, -0.0072065, 0.0071489
8: 0.9609597, 0.9861513, 0.9622233, 0.9849596, -0.0229544, 0.0222239
9: 0.0017233, 0.0091271, 0.0020735, 0.0087558, -0.0061399, 0.0062300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0138113
time: 0.92 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140194, upper bound: 0.0140289
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005189, 0.0007738, -0.0004492, 0.0006869, -0.0010627, 0.0010958
1: -0.0011087, 0.0025076, -0.0007826, 0.0023743, -0.0033982, 0.0031901
2: 0.0125846, 0.0180004, 0.0127842, 0.0175120, -0.0046166, 0.0049272
3: -0.0011638, 0.0029086, -0.0010138, 0.0025414, -0.0034076, 0.0036364
4: -0.0054531, -0.0016967, -0.0053147, -0.0020354, -0.0034177, 0.0036180
5: 0.0067765, 0.0108415, 0.0069263, 0.0104749, -0.0033957, 0.0036233
6: 0.0080754, 0.0103756, 0.0082774, 0.0103191, -0.0022437, 0.0020982
7: -0.0219351, -0.0131106, -0.0211394, -0.0134356, -0.0072653, 0.0068726
8: 0.9609442, 0.9862276, 0.9632240, 0.9852962, -0.0231634, 0.0217098
9: 0.0017009, 0.0091317, 0.0019746, 0.0084616, -0.0059231, 0.0062813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141411, upper bound: 0.0140092
time: 0.98 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141411, upper bound: 0.0142346
time: 1.00 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005184, 0.0007667, -0.0004697, 0.0006869, -0.0010690, 0.0011093
1: -0.0011065, 0.0024967, -0.0008785, 0.0023743, -0.0034184, 0.0032395
2: 0.0126010, 0.0179970, 0.0127842, 0.0176557, -0.0047080, 0.0049607
3: -0.0011515, 0.0029061, -0.0010137, 0.0026494, -0.0034819, 0.0036627
4: -0.0054418, -0.0016990, -0.0053147, -0.0019358, -0.0035060, 0.0036156
5: 0.0067888, 0.0108390, 0.0069263, 0.0105828, -0.0034705, 0.0036497
6: 0.0080919, 0.0103710, 0.0082775, 0.0103191, -0.0022271, 0.0020935
7: -0.0219297, -0.0131372, -0.0213734, -0.0134358, -0.0073332, 0.0070794
8: 0.9609597, 0.9861513, 0.9625535, 0.9852958, -0.0233175, 0.0221208
9: 0.0017233, 0.0091271, 0.0019747, 0.0086587, -0.0060898, 0.0063367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141693, upper bound: 0.0138391
time: 1.11 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141693, upper bound: 0.0140810
time: 1.06 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0005081, 0.0006743, -0.0004804, 0.0007054, -0.0010672, 0.0010106
1: -0.0010586, 0.0023550, -0.0009287, 0.0024027, -0.0031899, 0.0031159
2: 0.0128132, 0.0179253, 0.0127416, 0.0177307, -0.0045174, 0.0046684
3: -0.0009919, 0.0028522, -0.0010457, 0.0027059, -0.0033383, 0.0034662
4: -0.0052946, -0.0017488, -0.0053442, -0.0018837, -0.0034109, 0.0035954
5: 0.0069480, 0.0107851, 0.0068943, 0.0106391, -0.0033268, 0.0034558
6: 0.0083068, 0.0103109, 0.0082344, 0.0103312, -0.0020243, 0.0020765
7: -0.0218128, -0.0134829, -0.0214958, -0.0133663, -0.0070575, 0.0066788
8: 0.9612948, 0.9851607, 0.9622030, 0.9854949, -0.0219019, 0.0212317
9: 0.0020144, 0.0090287, 0.0019162, 0.0087617, -0.0057841, 0.0060801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148763, upper bound: 0.0145044
time: 0.86 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148763, upper bound: 0.0145044
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0004787, 0.0008152, -0.0004804, 0.0007054, -0.0010466, 0.0011497
1: -0.0009209, 0.0025710, -0.0009287, 0.0024027, -0.0031968, 0.0034591
2: 0.0124896, 0.0177192, 0.0127416, 0.0177307, -0.0050005, 0.0046372
3: -0.0012352, 0.0026972, -0.0010457, 0.0027059, -0.0036819, 0.0034262
4: -0.0055190, -0.0018918, -0.0053442, -0.0018837, -0.0036353, 0.0034525
5: 0.0067052, 0.0106304, 0.0068943, 0.0106391, -0.0036680, 0.0034144
6: 0.0079792, 0.0104025, 0.0082344, 0.0103312, -0.0023520, 0.0021682
7: -0.0214769, -0.0129557, -0.0214958, -0.0133663, -0.0068423, 0.0072757
8: 0.9622570, 0.9866712, 0.9622030, 0.9854949, -0.0217995, 0.0235260
9: 0.0015705, 0.0087458, 0.0019162, 0.0087617, -0.0063151, 0.0059264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148763, upper bound: 0.0145044
time: 0.89 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0148763, upper bound: 0.0145044
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0005076, 0.0007455, -0.0004804, 0.0007054, -0.0010527, 0.0010696
1: -0.0010561, 0.0024641, -0.0009287, 0.0024027, -0.0031751, 0.0031957
2: 0.0126497, 0.0179216, 0.0127416, 0.0177307, -0.0046205, 0.0046190
3: -0.0011148, 0.0028494, -0.0010457, 0.0027059, -0.0034083, 0.0034201
4: -0.0054080, -0.0017514, -0.0053442, -0.0018837, -0.0035242, 0.0035929
5: 0.0068254, 0.0107824, 0.0068943, 0.0106391, -0.0033960, 0.0034090
6: 0.0081413, 0.0103572, 0.0082344, 0.0103312, -0.0021899, 0.0021228
7: -0.0218067, -0.0132166, -0.0214958, -0.0133663, -0.0068962, 0.0068099
8: 0.9613120, 0.9859238, 0.9622030, 0.9854949, -0.0216941, 0.0217304
9: 0.0017901, 0.0090236, 0.0019162, 0.0087617, -0.0058908, 0.0059574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147286, upper bound: 0.0146648
time: 1.15 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147286, upper bound: 0.0146675
time: 1.22 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004793, 0.0008813, -0.0004804, 0.0007054, -0.0010336, 0.0012057
1: -0.0009236, 0.0026723, -0.0009287, 0.0024027, -0.0031869, 0.0035593
2: 0.0123380, 0.0177232, 0.0127416, 0.0177307, -0.0051398, 0.0046005
3: -0.0013492, 0.0027002, -0.0010457, 0.0027059, -0.0037823, 0.0033880
4: -0.0056242, -0.0018890, -0.0053442, -0.0018837, -0.0037404, 0.0034553
5: 0.0065914, 0.0106334, 0.0068943, 0.0106391, -0.0037676, 0.0033756
6: 0.0078256, 0.0104455, 0.0082344, 0.0103312, -0.0025055, 0.0022111
7: -0.0214834, -0.0127087, -0.0214958, -0.0133663, -0.0066988, 0.0074280
8: 0.9622383, 0.9873791, 0.9622030, 0.9854949, -0.0216472, 0.0241891
9: 0.0013624, 0.0087514, 0.0019162, 0.0087617, -0.0064638, 0.0058193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147286, upper bound: 0.0146648
time: 0.96 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147286, upper bound: 0.0146675
time: 1.18 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0005180, 0.0006762, -0.0004641, 0.0007781, -0.0011549, 0.0010171
1: -0.0011048, 0.0023580, -0.0008525, 0.0025141, -0.0034745, 0.0031714
2: 0.0128086, 0.0179946, 0.0125748, 0.0176167, -0.0045710, 0.0050603
3: -0.0009954, 0.0029043, -0.0011711, 0.0026201, -0.0033662, 0.0037451
4: -0.0052977, -0.0017007, -0.0054599, -0.0019629, -0.0033349, 0.0037592
5: 0.0069446, 0.0108372, 0.0067692, 0.0105535, -0.0033538, 0.0037327
6: 0.0083022, 0.0103122, 0.0080655, 0.0103784, -0.0020762, 0.0022467
7: -0.0219257, -0.0134755, -0.0213099, -0.0130946, -0.0075878, 0.0067334
8: 0.9609711, 0.9851819, 0.9627355, 0.9862733, -0.0237666, 0.0215174
9: 0.0020082, 0.0091238, 0.0016874, 0.0086052, -0.0058154, 0.0065303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142231, upper bound: 0.0139908
time: 1.01 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142467, upper bound: 0.0138113
time: 0.91 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0005180, 0.0006762, -0.0004502, 0.0008016, -0.0011774, 0.0010052
1: -0.0011048, 0.0023580, -0.0007871, 0.0025501, -0.0035129, 0.0031451
2: 0.0128086, 0.0179946, 0.0125209, 0.0175188, -0.0045226, 0.0051178
3: -0.0009954, 0.0029043, -0.0012117, 0.0025465, -0.0033264, 0.0037883
4: -0.0052977, -0.0017007, -0.0054973, -0.0020307, -0.0032670, 0.0037966
5: 0.0069446, 0.0108372, 0.0067286, 0.0104800, -0.0033139, 0.0037759
6: 0.0083022, 0.0103122, 0.0080108, 0.0103937, -0.0020915, 0.0023014
7: -0.0219257, -0.0134755, -0.0211504, -0.0130066, -0.0076815, 0.0066059
8: 0.9609711, 0.9851819, 0.9631923, 0.9865254, -0.0240350, 0.0212989
9: 0.0020082, 0.0091238, 0.0016133, 0.0084709, -0.0057226, 0.0066092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142231, upper bound: 0.0140062
time: 0.97 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0142467, upper bound: 0.0138391
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0005177, 0.0007474, -0.0004641, 0.0007781, -0.0011415, 0.0010708
1: -0.0011034, 0.0024670, -0.0008525, 0.0025141, -0.0034736, 0.0032467
2: 0.0126454, 0.0179924, 0.0125748, 0.0176167, -0.0046630, 0.0050349
3: -0.0011181, 0.0029026, -0.0011711, 0.0026201, -0.0034250, 0.0037158
4: -0.0054110, -0.0017023, -0.0054599, -0.0019629, -0.0034481, 0.0037576
5: 0.0068221, 0.0108355, 0.0067692, 0.0105535, -0.0034116, 0.0037025
6: 0.0081370, 0.0103584, 0.0080655, 0.0103784, -0.0022415, 0.0022929
7: -0.0219221, -0.0132096, -0.0213099, -0.0130946, -0.0074385, 0.0067983
8: 0.9609815, 0.9859439, 0.9627355, 0.9862733, -0.0236724, 0.0219683
9: 0.0017842, 0.0091207, 0.0016874, 0.0086052, -0.0058739, 0.0064257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141367, upper bound: 0.0142067
time: 1.10 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141689, upper bound: 0.0140519
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0005177, 0.0007474, -0.0004502, 0.0008016, -0.0011645, 0.0010625
1: -0.0011034, 0.0024670, -0.0007871, 0.0025501, -0.0035154, 0.0032402
2: 0.0126454, 0.0179924, 0.0125209, 0.0175188, -0.0046397, 0.0050976
3: -0.0011181, 0.0029026, -0.0012117, 0.0025465, -0.0034014, 0.0037629
4: -0.0054110, -0.0017023, -0.0054973, -0.0020307, -0.0033802, 0.0037951
5: 0.0068221, 0.0108355, 0.0067286, 0.0104800, -0.0033876, 0.0037495
6: 0.0081370, 0.0103584, 0.0080108, 0.0103937, -0.0022567, 0.0023476
7: -0.0219221, -0.0132096, -0.0211504, -0.0130066, -0.0075407, 0.0067172
8: 0.9609815, 0.9859439, 0.9631923, 0.9865254, -0.0239650, 0.0218717
9: 0.0017842, 0.0091207, 0.0016133, 0.0084709, -0.0058223, 0.0065117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141367, upper bound: 0.0142395
time: 1.08 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0141689, upper bound: 0.0140950
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0004987, 0.0007125, -0.0004919, 0.0006571, -0.0010074, 0.0010614
1: -0.0010142, 0.0024136, -0.0009825, 0.0023287, -0.0030743, 0.0031717
2: 0.0127254, 0.0178589, 0.0128525, 0.0178115, -0.0046104, 0.0044837
3: -0.0010579, 0.0028022, -0.0009624, 0.0027666, -0.0034136, 0.0033227
4: -0.0053555, -0.0017949, -0.0052673, -0.0018278, -0.0035277, 0.0034725
5: 0.0068821, 0.0107353, 0.0069775, 0.0106997, -0.0034026, 0.0033121
6: 0.0082179, 0.0103358, 0.0083466, 0.0102998, -0.0020818, 0.0019892
7: -0.0217045, -0.0133399, -0.0216273, -0.0135469, -0.0067225, 0.0069398
8: 0.9616048, 0.9855705, 0.9618262, 0.9849774, -0.0210482, 0.0216611
9: 0.0018939, 0.0089375, 0.0020683, 0.0088725, -0.0059716, 0.0057995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146274, upper bound: 0.0141789
time: 0.87 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B1_A1_A2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144610, upper bound: 0.0142299
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004682, 0.0008471, -0.0004919, 0.0006571, -0.0009847, 0.0011895
1: -0.0008718, 0.0026199, -0.0009825, 0.0023287, -0.0030639, 0.0034835
2: 0.0124164, 0.0176455, 0.0128525, 0.0178115, -0.0050630, 0.0044262
3: -0.0012903, 0.0026418, -0.0009624, 0.0027666, -0.0037401, 0.0032622
4: -0.0055698, -0.0019428, -0.0052673, -0.0018278, -0.0037421, 0.0033245
5: 0.0066502, 0.0105752, 0.0069775, 0.0106997, -0.0037271, 0.0032502
6: 0.0079050, 0.0104233, 0.0083466, 0.0102998, -0.0023948, 0.0020767
7: -0.0213569, -0.0128364, -0.0216273, -0.0135469, -0.0064793, 0.0074758
8: 0.9626008, 0.9870131, 0.9618262, 0.9849774, -0.0208251, 0.0237937
9: 0.0014700, 0.0086448, 0.0020683, 0.0088725, -0.0064685, 0.0056230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0146274, upper bound: 0.0141789
time: 0.92 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0144610, upper bound: 0.0142299
time: 1.04 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0005081, 0.0007145, -0.0004620, 0.0007002, -0.0010676, 0.0010493
1: -0.0010582, 0.0024167, -0.0008424, 0.0023947, -0.0033306, 0.0031750
2: 0.0127207, 0.0179248, 0.0127537, 0.0176015, -0.0045881, 0.0048380
3: -0.0010614, 0.0028518, -0.0010367, 0.0026087, -0.0033829, 0.0035740
4: -0.0053587, -0.0017491, -0.0053359, -0.0019734, -0.0033853, 0.0035867
5: 0.0068787, 0.0107848, 0.0069034, 0.0105421, -0.0033706, 0.0035615
6: 0.0082132, 0.0103371, 0.0082466, 0.0103278, -0.0021145, 0.0020905
7: -0.0218119, -0.0133323, -0.0212852, -0.0133860, -0.0071801, 0.0067888
8: 0.9612970, 0.9855922, 0.9628063, 0.9854385, -0.0227350, 0.0215822
9: 0.0018876, 0.0090280, 0.0019328, 0.0085844, -0.0058542, 0.0061995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A2_A1_B1_B2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140570, upper bound: 0.0141354
time: 1.04 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B2_B1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140802, upper bound: 0.0139471
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0005081, 0.0007145, -0.0004641, 0.0007781, -0.0011472, 0.0010571
1: -0.0010582, 0.0024167, -0.0008525, 0.0025141, -0.0034629, 0.0032351
2: 0.0127207, 0.0179248, 0.0125748, 0.0176167, -0.0046664, 0.0050361
3: -0.0010614, 0.0028518, -0.0011711, 0.0026201, -0.0034379, 0.0037230
4: -0.0053587, -0.0017491, -0.0054599, -0.0019629, -0.0033959, 0.0037108
5: 0.0068787, 0.0107848, 0.0067692, 0.0105535, -0.0034254, 0.0037102
6: 0.0082132, 0.0103371, 0.0080655, 0.0103784, -0.0021652, 0.0022716
7: -0.0218119, -0.0133323, -0.0213099, -0.0130946, -0.0075029, 0.0068889
8: 0.9612970, 0.9855922, 0.9627355, 0.9862733, -0.0236601, 0.0219628
9: 0.0018876, 0.0090280, 0.0016874, 0.0086052, -0.0059464, 0.0064714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_A2_A1_B1_B2_B2_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140570, upper bound: 0.0141354
time: 0.99 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B2_B2_B2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0140802, upper bound: 0.0139471
time: 1.16 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0004987, 0.0007125, -0.0004804, 0.0006905, -0.0010449, 0.0010537
1: -0.0010142, 0.0024136, -0.0009287, 0.0023799, -0.0031779, 0.0032106
2: 0.0127254, 0.0178589, 0.0127759, 0.0177308, -0.0046477, 0.0046316
3: -0.0010579, 0.0028022, -0.0010200, 0.0027059, -0.0034317, 0.0034319
4: -0.0053555, -0.0017949, -0.0053205, -0.0018837, -0.0034718, 0.0035256
5: 0.0068821, 0.0107353, 0.0069200, 0.0106392, -0.0034197, 0.0034210
6: 0.0082179, 0.0103358, 0.0082691, 0.0103215, -0.0021035, 0.0020667
7: -0.0217045, -0.0133399, -0.0214959, -0.0134221, -0.0069323, 0.0069025
8: 0.9616048, 0.9855705, 0.9622025, 0.9853348, -0.0217445, 0.0218571
9: 0.0018939, 0.0089375, 0.0019632, 0.0087618, -0.0059575, 0.0059886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150169, upper bound: 0.0146768
time: 1.02 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150169, upper bound: 0.0146768
time: 1.02 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0004987, 0.0007125, -0.0004514, 0.0008248, -0.0011853, 0.0010423
1: -0.0010142, 0.0024136, -0.0007930, 0.0025856, -0.0035302, 0.0032066
2: 0.0127254, 0.0178589, 0.0124677, 0.0175276, -0.0046937, 0.0051248
3: -0.0010579, 0.0028022, -0.0012517, 0.0025531, -0.0034506, 0.0037888
4: -0.0053555, -0.0017949, -0.0055342, -0.0020246, -0.0033309, 0.0037393
5: 0.0068821, 0.0107353, 0.0066887, 0.0104866, -0.0034373, 0.0037759
6: 0.0082179, 0.0103358, 0.0079570, 0.0104088, -0.0021908, 0.0023788
7: -0.0217045, -0.0133399, -0.0211648, -0.0129200, -0.0076255, 0.0068006
8: 0.9616048, 0.9855705, 0.9631513, 0.9867736, -0.0240885, 0.0221009
9: 0.0018939, 0.0089375, 0.0015404, 0.0084830, -0.0059167, 0.0065859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 173

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of IS_B1_A2_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150169, upper bound: 0.0146768
time: 1.02 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0150169, upper bound: 0.0146768
time: 1.02 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0004658, 0.0008015, -0.0004897, 0.0006926, -0.0010354, 0.0011546
1: -0.0008604, 0.0025499, -0.0009724, 0.0023831, -0.0031996, 0.0035111
2: 0.0125212, 0.0176286, 0.0127710, 0.0177962, -0.0050963, 0.0046295
3: -0.0012115, 0.0026290, -0.0010236, 0.0027551, -0.0037612, 0.0034160
4: -0.0054971, -0.0019546, -0.0053238, -0.0018383, -0.0036588, 0.0033692
5: 0.0067289, 0.0105624, 0.0069164, 0.0106882, -0.0037476, 0.0034038
6: 0.0080112, 0.0103936, 0.0082641, 0.0103228, -0.0023116, 0.0021294
7: -0.0213293, -0.0130072, -0.0216024, -0.0134142, -0.0068130, 0.0074600
8: 0.9626800, 0.9865237, 0.9618973, 0.9853576, -0.0217700, 0.0239563
9: 0.0016138, 0.0086215, 0.0019566, 0.0088515, -0.0064779, 0.0059075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_B1_A2_A2_A1_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147923, upper bound: 0.0141613
time: 1.03 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0147923, upper bound: 0.0141613
time: 1.16 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0004867, 0.0008022, -0.0004892, 0.0006851, -0.0010464, 0.0011601
1: -0.0009584, 0.0025510, -0.0009700, 0.0023716, -0.0032450, 0.0035211
2: 0.0125195, 0.0177752, 0.0127883, 0.0177927, -0.0051279, 0.0047167
3: -0.0012127, 0.0027393, -0.0010106, 0.0027525, -0.0037860, 0.0034888
4: -0.0054983, -0.0018529, -0.0053119, -0.0018407, -0.0036575, 0.0034590
5: 0.0067276, 0.0106725, 0.0069294, 0.0106856, -0.0037725, 0.0034770
6: 0.0080095, 0.0103941, 0.0082816, 0.0103179, -0.0023085, 0.0021124
7: -0.0215682, -0.0130045, -0.0215968, -0.0134424, -0.0069996, 0.0075259
8: 0.9619954, 0.9865316, 0.9619135, 0.9852769, -0.0221547, 0.0241014
9: 0.0016115, 0.0088227, 0.0019803, 0.0088468, -0.0065309, 0.0060569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.16 + 597.60 = 600.76 seconds
