## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00070371


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0057997, 0.0064849, 0.0057997, 0.0064849, -0.0006851, 0.0006851)
1: (-0.0010220, 0.0006427, -0.0010220, 0.0006427, -0.0016647, 0.0016647)
2: (0.0117443, 0.0228718, 0.0117443, 0.0228718, -0.0111275, 0.0111275)
3: (-0.0045175, -0.0035615, -0.0045175, -0.0035615, -0.0009560, 0.0009560)
4: (-0.0005819, 0.0040564, -0.0005819, 0.0040564, -0.0046383, 0.0046383)
5: (-0.0011253, -0.0001563, -0.0011253, -0.0001563, -0.0009690, 0.0009690)
6: (0.9903233, 0.9923169, 0.9903233, 0.9923169, -0.0019936, 0.0019936)
7: (-0.0145760, -0.0060401, -0.0145760, -0.0060401, -0.0066597, 0.0066597)
8: (-0.0055986, -0.0009040, -0.0055986, -0.0009040, -0.0046946, 0.0046946)
9: (-0.0055249, -0.0000012, -0.0055249, -0.0000012, -0.0055237, 0.0055237)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 2.11 = 3.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0012715, upper bound: 0.0012715

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0012198, upper bound: 0.0012281
time: 1.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0012281, upper bound: 0.0012281
time: 1.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.48 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 6, lower bound: -0.0012198, upper bound: 0.0012281
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.48
Output dim: 6, lower bound: -0.0012281, upper bound: 0.0012281

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0058205, 0.0064840, 0.0058049, 0.0064846, -0.0006641, 0.0006790
1: -0.0008384, 0.0006409, -0.0009762, 0.0006422, -0.0014806, 0.0016171
2: 0.0117583, 0.0223671, 0.0117477, 0.0227459, -0.0109876, 0.0106194
3: -0.0044885, -0.0035628, -0.0045103, -0.0035618, -0.0009267, 0.0009475
4: -0.0004413, 0.0040503, -0.0005469, 0.0040549, -0.0044962, 0.0045972
5: -0.0011244, -0.0002948, -0.0011250, -0.0001908, -0.0009335, 0.0008303
6: 0.9906692, 0.9923152, 0.9904097, 0.9923165, -0.0016473, 0.0019055
7: -0.0142620, -0.0060511, -0.0144977, -0.0060428, -0.0063617, 0.0065540
8: -0.0046420, -0.0009074, -0.0053601, -0.0009048, -0.0037372, 0.0044526
9: -0.0055180, -0.0002767, -0.0055232, -0.0000699, -0.0054481, 0.0052466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011778, upper bound: 0.0011799
time: 0.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011767, upper bound: 0.0011831
time: 1.01 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0058253, 0.0065044, 0.0058110, 0.0064843, -0.0006589, 0.0006934
1: -0.0007958, 0.0006804, -0.0009227, 0.0006415, -0.0014373, 0.0016031
2: 0.0114396, 0.0222502, 0.0117539, 0.0225988, -0.0111593, 0.0104963
3: -0.0044818, -0.0035343, -0.0045018, -0.0035624, -0.0009194, 0.0009675
4: -0.0004087, 0.0041884, -0.0005059, 0.0040522, -0.0044609, 0.0046943
5: -0.0011450, -0.0003269, -0.0011246, -0.0002312, -0.0009138, 0.0007978
6: 0.9907494, 0.9923530, 0.9905105, 0.9923157, -0.0015664, 0.0018426
7: -0.0141892, -0.0058011, -0.0144062, -0.0060477, -0.0063983, 0.0067517
8: -0.0044204, -0.0008291, -0.0050812, -0.0009063, -0.0035141, 0.0042521
9: -0.0056744, -0.0003405, -0.0055202, -0.0001502, -0.0055242, 0.0051797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011849, upper bound: 0.0011799
time: 1.09 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011831, upper bound: 0.0011831
time: 1.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.85 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 6, lower bound: -0.0011778, upper bound: 0.0011799
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 6, lower bound: -0.0011767, upper bound: 0.0011831
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 6, lower bound: -0.0011849, upper bound: 0.0011799
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.85
Output dim: 6, lower bound: -0.0011831, upper bound: 0.0011831

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058239, 0.0064838, 0.0058205, 0.0064838, -0.0006599, 0.0006633
1: -0.0008087, 0.0006405, -0.0008385, 0.0006405, -0.0014493, 0.0014790
2: 0.0117613, 0.0222857, 0.0117614, 0.0223674, -0.0106061, 0.0105243
3: -0.0044838, -0.0035630, -0.0044885, -0.0035630, -0.0009208, 0.0009255
4: -0.0004186, 0.0040490, -0.0004414, 0.0040490, -0.0044675, 0.0044903
5: -0.0011242, -0.0003171, -0.0011242, -0.0002947, -0.0008295, 0.0008071
6: 0.9907251, 0.9923149, 0.9906690, 0.9923148, -0.0015897, 0.0016459
7: -0.0142113, -0.0060535, -0.0142621, -0.0060535, -0.0062896, 0.0063327
8: -0.0044877, -0.0009082, -0.0046426, -0.0009082, -0.0035795, 0.0037344
9: -0.0055166, -0.0003211, -0.0055165, -0.0002765, -0.0052400, 0.0051954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010917, upper bound: 0.0010803
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011514, upper bound: 0.0011534
time: 1.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058299, 0.0064832, 0.0058255, 0.0065014, -0.0006715, 0.0006577
1: -0.0007553, 0.0006394, -0.0007941, 0.0006748, -0.0014301, 0.0014335
2: 0.0117707, 0.0221389, 0.0114850, 0.0222455, -0.0104747, 0.0106539
3: -0.0044754, -0.0035639, -0.0044815, -0.0035384, -0.0009370, 0.0009177
4: -0.0003777, 0.0040449, -0.0004074, 0.0041687, -0.0045464, 0.0044523
5: -0.0011236, -0.0003574, -0.0011420, -0.0003282, -0.0007954, 0.0007847
6: 0.9908257, 0.9923137, 0.9907526, 0.9923477, -0.0015219, 0.0015611
7: -0.0141200, -0.0060608, -0.0141863, -0.0058367, -0.0063989, 0.0063768
8: -0.0042095, -0.0009105, -0.0044114, -0.0008403, -0.0033692, 0.0035010
9: -0.0055119, -0.0004013, -0.0056521, -0.0003431, -0.0051689, 0.0052508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010744, upper bound: 0.0010933
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011507, upper bound: 0.0011566
time: 1.21 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058287, 0.0065041, 0.0058266, 0.0064833, -0.0006547, 0.0006776
1: -0.0007661, 0.0006801, -0.0007846, 0.0006397, -0.0014058, 0.0014647
2: 0.0114426, 0.0221686, 0.0117679, 0.0222195, -0.0107769, 0.0104007
3: -0.0044771, -0.0035346, -0.0044800, -0.0035636, -0.0009135, 0.0009455
4: -0.0003859, 0.0041871, -0.0004001, 0.0040461, -0.0044321, 0.0045872
5: -0.0011448, -0.0003493, -0.0011237, -0.0003353, -0.0008095, 0.0007745
6: 0.9908051, 0.9923528, 0.9907705, 0.9923141, -0.0015090, 0.0015823
7: -0.0141385, -0.0058035, -0.0141701, -0.0060587, -0.0063266, 0.0065206
8: -0.0042658, -0.0008298, -0.0043622, -0.0009098, -0.0033560, 0.0035324
9: -0.0056729, -0.0003850, -0.0055133, -0.0003573, -0.0053156, 0.0051283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010964, upper bound: 0.0010803
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011582, upper bound: 0.0011534
time: 1.06 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058347, 0.0065036, 0.0058312, 0.0065011, -0.0006663, 0.0006723
1: -0.0007125, 0.0006790, -0.0007435, 0.0006741, -0.0013866, 0.0014224
2: 0.0114515, 0.0220212, 0.0114906, 0.0221064, -0.0106549, 0.0105306
3: -0.0044686, -0.0035354, -0.0044735, -0.0035389, -0.0009298, 0.0009382
4: -0.0003449, 0.0041832, -0.0003686, 0.0041663, -0.0045112, 0.0045518
5: -0.0011442, -0.0003897, -0.0011417, -0.0003663, -0.0007779, 0.0007520
6: 0.9909063, 0.9923517, 0.9908479, 0.9923471, -0.0014408, 0.0015037
7: -0.0140468, -0.0058105, -0.0140998, -0.0058411, -0.0064383, 0.0065670
8: -0.0039865, -0.0008320, -0.0041479, -0.0008416, -0.0031448, 0.0033158
9: -0.0056685, -0.0004655, -0.0056493, -0.0004190, -0.0052495, 0.0051839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010933, upper bound: 0.0010839
time: 1.16 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011566, upper bound: 0.0011566
time: 1.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.92 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0010917, upper bound: 0.0010803
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0011514, upper bound: 0.0011534
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0010744, upper bound: 0.0010933
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0011507, upper bound: 0.0011566
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0010964, upper bound: 0.0010803
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0011582, upper bound: 0.0011534
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0010933, upper bound: 0.0010839
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 6, lower bound: -0.0011566, upper bound: 0.0011566

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058384, 0.0064828, 0.0058590, 0.0065141, -0.0003946, 0.0006238
1: -0.0006799, 0.0006387, -0.0005695, 0.0006992, -0.0013791, 0.0006951
2: 0.0117761, 0.0219316, 0.0112880, 0.0215217, -0.0056064, 0.0106436
3: -0.0044635, -0.0035644, -0.0044348, -0.0035208, -0.0005506, 0.0008704
4: -0.0003199, 0.0040426, -0.0001806, 0.0042541, -0.0026713, 0.0042232
5: -0.0011232, -0.0004143, -0.0011548, -0.0004928, -0.0003627, 0.0007405
6: 0.9909676, 0.9923131, 0.9911569, 0.9923710, -0.0014034, 0.0006652
7: -0.0139910, -0.0060651, -0.0137097, -0.0056822, -0.0062992, 0.0043977
8: -0.0038166, -0.0009118, -0.0033068, -0.0007919, -0.0030247, 0.0013778
9: -0.0055093, -0.0005144, -0.0057487, -0.0007292, -0.0027499, 0.0052343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010606, upper bound: 0.0010429
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010729, upper bound: 0.0010650
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058239, 0.0064838, 0.0058301, 0.0064834, -0.0006595, 0.0006537
1: -0.0008087, 0.0006405, -0.0007538, 0.0006398, -0.0014485, 0.0013944
2: 0.0117613, 0.0222857, 0.0117675, 0.0221349, -0.0103736, 0.0105182
3: -0.0044838, -0.0035630, -0.0044752, -0.0035636, -0.0009202, 0.0009121
4: -0.0004186, 0.0040490, -0.0003765, 0.0040463, -0.0044649, 0.0044255
5: -0.0011242, -0.0003171, -0.0011238, -0.0003585, -0.0007657, 0.0008067
6: 0.9907251, 0.9923149, 0.9908283, 0.9923142, -0.0015891, 0.0014865
7: -0.0142113, -0.0060535, -0.0141175, -0.0060583, -0.0062813, 0.0061263
8: -0.0044877, -0.0009082, -0.0042019, -0.0009097, -0.0035780, 0.0032937
9: -0.0055166, -0.0003211, -0.0055135, -0.0004034, -0.0051131, 0.0051924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011086, upper bound: 0.0011165
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011366, upper bound: 0.0011383
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058692, 0.0065138, 0.0058396, 0.0065006, -0.0006314, 0.0003922
1: -0.0005499, 0.0006987, -0.0006695, 0.0006731, -0.0007239, 0.0013682
2: 0.0112921, 0.0213631, 0.0114987, 0.0219031, -0.0106110, 0.0058387
3: -0.0044206, -0.0035211, -0.0044619, -0.0035396, -0.0008810, 0.0005472
4: -0.0001118, 0.0042523, -0.0003119, 0.0041628, -0.0042746, 0.0026550
5: -0.0011545, -0.0005030, -0.0011412, -0.0004221, -0.0007324, 0.0003777
6: 0.9911757, 0.9923705, 0.9909872, 0.9923460, -0.0006927, 0.0013833
7: -0.0135853, -0.0056854, -0.0139733, -0.0058475, -0.0045800, 0.0064081
8: -0.0032678, -0.0007929, -0.0037626, -0.0008436, -0.0014349, 0.0029697
9: -0.0057467, -0.0008070, -0.0056454, -0.0005300, -0.0052167, 0.0028638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010407, upper bound: 0.0010611
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010595, upper bound: 0.0010738
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058393, 0.0064828, 0.0058255, 0.0065014, -0.0006621, 0.0006573
1: -0.0006723, 0.0006386, -0.0007941, 0.0006748, -0.0013471, 0.0014327
2: 0.0117768, 0.0219107, 0.0114850, 0.0222455, -0.0104687, 0.0104257
3: -0.0044623, -0.0035644, -0.0044815, -0.0035384, -0.0009239, 0.0009171
4: -0.0003140, 0.0040423, -0.0004074, 0.0041687, -0.0044828, 0.0044496
5: -0.0011232, -0.0004200, -0.0011420, -0.0003282, -0.0007950, 0.0007220
6: 0.9909821, 0.9923130, 0.9907526, 0.9923477, -0.0013656, 0.0015604
7: -0.0139780, -0.0060656, -0.0141863, -0.0058367, -0.0062618, 0.0063687
8: -0.0037769, -0.0009120, -0.0044114, -0.0008403, -0.0029367, 0.0034995
9: -0.0055090, -0.0005258, -0.0056521, -0.0003431, -0.0051659, 0.0051263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011126, upper bound: 0.0011139
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011360, upper bound: 0.0011419
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058428, 0.0065032, 0.0058644, 0.0065138, -0.0003884, 0.0006388
1: -0.0006416, 0.0006783, -0.0005590, 0.0006988, -0.0013404, 0.0007446
2: 0.0114572, 0.0218265, 0.0112913, 0.0214369, -0.0060059, 0.0105352
3: -0.0044575, -0.0035359, -0.0044272, -0.0035210, -0.0005419, 0.0008913
4: -0.0002906, 0.0041808, -0.0001438, 0.0042527, -0.0026294, 0.0043246
5: -0.0011438, -0.0004431, -0.0011546, -0.0004983, -0.0003885, 0.0007115
6: 0.9910397, 0.9923510, 0.9911669, 0.9923707, -0.0013310, 0.0007126
7: -0.0139256, -0.0058149, -0.0136432, -0.0056848, -0.0063428, 0.0047111
8: -0.0036174, -0.0008334, -0.0032860, -0.0007927, -0.0028247, 0.0014760
9: -0.0056657, -0.0005718, -0.0057471, -0.0007708, -0.0029458, 0.0051753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010628, upper bound: 0.0010427
time: 1.19 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010777, upper bound: 0.0010650
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058287, 0.0065041, 0.0058362, 0.0064829, -0.0006542, 0.0006680
1: -0.0007661, 0.0006801, -0.0006999, 0.0006389, -0.0014051, 0.0013800
2: 0.0114426, 0.0221686, 0.0117743, 0.0219868, -0.0105441, 0.0103943
3: -0.0044771, -0.0035346, -0.0044667, -0.0035642, -0.0009129, 0.0009321
4: -0.0003859, 0.0041871, -0.0003352, 0.0040433, -0.0044293, 0.0045223
5: -0.0011448, -0.0003493, -0.0011233, -0.0003991, -0.0007456, 0.0007741
6: 0.9908051, 0.9923528, 0.9909299, 0.9923134, -0.0015082, 0.0014229
7: -0.0141385, -0.0058035, -0.0140253, -0.0060637, -0.0063182, 0.0064401
8: -0.0042658, -0.0008298, -0.0039211, -0.0009114, -0.0033544, 0.0030912
9: -0.0056729, -0.0003850, -0.0055102, -0.0004843, -0.0051886, 0.0051251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011214, upper bound: 0.0011108
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011434, upper bound: 0.0011383
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058487, 0.0065026, 0.0058693, 0.0065320, -0.0004169, 0.0003875
1: -0.0005895, 0.0006771, -0.0005496, 0.0007341, -0.0008075, 0.0007505
2: 0.0114663, 0.0216826, 0.0110070, 0.0213611, -0.0060539, 0.0065133
3: -0.0044491, -0.0035367, -0.0044204, -0.0034957, -0.0005817, 0.0005407
4: -0.0002503, 0.0041768, -0.0001110, 0.0043759, -0.0028225, 0.0026234
5: -0.0011433, -0.0004824, -0.0011730, -0.0005032, -0.0003916, 0.0004213
6: 0.9911377, 0.9923499, 0.9911759, 0.9924043, -0.0007728, 0.0007182
7: -0.0138359, -0.0058221, -0.0135837, -0.0054618, -0.0051092, 0.0047488
8: -0.0033463, -0.0008357, -0.0032673, -0.0007228, -0.0016007, 0.0014878
9: -0.0056613, -0.0006503, -0.0058866, -0.0008080, -0.0029694, 0.0031947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010627, upper bound: 0.0010560
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010738, upper bound: 0.0010694
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058347, 0.0065036, 0.0058407, 0.0065007, -0.0006659, 0.0006629
1: -0.0007125, 0.0006790, -0.0006597, 0.0006733, -0.0013858, 0.0013387
2: 0.0114515, 0.0220212, 0.0114969, 0.0218763, -0.0104247, 0.0105244
3: -0.0044686, -0.0035354, -0.0044603, -0.0035394, -0.0009292, 0.0009249
4: -0.0003449, 0.0041832, -0.0003044, 0.0041636, -0.0045085, 0.0044877
5: -0.0011442, -0.0003897, -0.0011413, -0.0004295, -0.0007148, 0.0007516
6: 0.9909063, 0.9923517, 0.9910057, 0.9923462, -0.0014399, 0.0013459
7: -0.0140468, -0.0058105, -0.0139566, -0.0058460, -0.0064298, 0.0064846
8: -0.0039865, -0.0008320, -0.0037117, -0.0008432, -0.0031433, 0.0028796
9: -0.0056685, -0.0004655, -0.0056463, -0.0005446, -0.0051239, 0.0051808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011139, upper bound: 0.0011199
time: 1.35 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0011419, upper bound: 0.0011419
time: 1.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.31 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010606, upper bound: 0.0010429
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010729, upper bound: 0.0010650
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011086, upper bound: 0.0011165
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011366, upper bound: 0.0011383
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010407, upper bound: 0.0010611
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010595, upper bound: 0.0010738
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011126, upper bound: 0.0011139
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011360, upper bound: 0.0011419
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010628, upper bound: 0.0010427
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010777, upper bound: 0.0010650
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011214, upper bound: 0.0011108
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011434, upper bound: 0.0011383
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010627, upper bound: 0.0010560
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0010738, upper bound: 0.0010694
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011139, upper bound: 0.0011199
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.31
Output dim: 6, lower bound: -0.0011419, upper bound: 0.0011419

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058283, 0.0064556, 0.0058617, 0.0065020, -0.0003774, 0.0005939
1: -0.0007694, 0.0005859, -0.0005643, 0.0006759, -0.0014454, 0.0006411
2: 0.0122018, 0.0221777, 0.0114760, 0.0214797, -0.0051714, 0.0107017
3: -0.0044776, -0.0036024, -0.0044310, -0.0035375, -0.0005267, 0.0008287
4: -0.0003885, 0.0038581, -0.0001624, 0.0041726, -0.0025553, 0.0040205
5: -0.0010957, -0.0003467, -0.0011426, -0.0004955, -0.0003345, 0.0007959
6: 0.9907990, 0.9922627, 0.9911619, 0.9923487, -0.0015497, 0.0006136
7: -0.0141441, -0.0063990, -0.0136768, -0.0058296, -0.0061052, 0.0040566
8: -0.0042830, -0.0010164, -0.0032965, -0.0008380, -0.0034449, 0.0012709
9: -0.0053005, -0.0003801, -0.0056565, -0.0007498, -0.0025365, 0.0052764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010242, upper bound: 0.0009998
time: 1.10 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010193, upper bound: 0.0009996
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058400, 0.0064741, 0.0058591, 0.0065136, -0.0003926, 0.0006150
1: -0.0006658, 0.0006218, -0.0005693, 0.0006983, -0.0013641, 0.0006677
2: 0.0119125, 0.0218929, 0.0112956, 0.0215199, -0.0053856, 0.0105973
3: -0.0044613, -0.0035765, -0.0044346, -0.0035214, -0.0005478, 0.0008581
4: -0.0003091, 0.0039835, -0.0001798, 0.0042508, -0.0026580, 0.0041633
5: -0.0011144, -0.0004249, -0.0011543, -0.0004929, -0.0003484, 0.0007294
6: 0.9909942, 0.9922969, 0.9911571, 0.9923701, -0.0013759, 0.0006390
7: -0.0139669, -0.0061720, -0.0137083, -0.0056882, -0.0062306, 0.0042246
8: -0.0037432, -0.0009453, -0.0033064, -0.0007937, -0.0029495, 0.0013235
9: -0.0054424, -0.0005355, -0.0057450, -0.0007301, -0.0026416, 0.0052094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010331, upper bound: 0.0010168
time: 1.16 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010275, upper bound: 0.0010161
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058266, 0.0064735, 0.0058182, 0.0064560, -0.0006295, 0.0006553
1: -0.0007845, 0.0006207, -0.0008585, 0.0005869, -0.0013714, 0.0014793
2: 0.0119213, 0.0222193, 0.0121943, 0.0224225, -0.0105012, 0.0100249
3: -0.0044800, -0.0035773, -0.0044917, -0.0036017, -0.0008783, 0.0009144
4: -0.0004001, 0.0039797, -0.0004567, 0.0038614, -0.0042614, 0.0044364
5: -0.0011138, -0.0003353, -0.0010962, -0.0002796, -0.0008342, 0.0007608
6: 0.9907706, 0.9922959, 0.9906313, 0.9922635, -0.0014929, 0.0016646
7: -0.0141700, -0.0061790, -0.0142965, -0.0063931, -0.0058811, 0.0058466
8: -0.0043618, -0.0009475, -0.0047470, -0.0010146, -0.0033472, 0.0037996
9: -0.0054381, -0.0003574, -0.0053042, -0.0002464, -0.0051917, 0.0049468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010751, upper bound: 0.0010760
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010712, upper bound: 0.0010758
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058240, 0.0064832, 0.0058317, 0.0064747, -0.0006507, 0.0006515
1: -0.0008078, 0.0006394, -0.0007398, 0.0006229, -0.0014307, 0.0013792
2: 0.0117707, 0.0222831, 0.0119036, 0.0220963, -0.0103256, 0.0103795
3: -0.0044837, -0.0035639, -0.0044730, -0.0035757, -0.0009079, 0.0009091
4: -0.0004179, 0.0040449, -0.0003658, 0.0039873, -0.0044052, 0.0044107
5: -0.0011236, -0.0003178, -0.0011150, -0.0003691, -0.0007545, 0.0007971
6: 0.9907269, 0.9923137, 0.9908548, 0.9922979, -0.0015711, 0.0014589
7: -0.0142097, -0.0060608, -0.0140935, -0.0061651, -0.0061945, 0.0060571
8: -0.0044828, -0.0009105, -0.0041287, -0.0009431, -0.0035396, 0.0032183
9: -0.0055120, -0.0003225, -0.0054468, -0.0004245, -0.0050875, 0.0051242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010934, upper bound: 0.0010898
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010894
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058719, 0.0065018, 0.0058295, 0.0064738, -0.0006019, 0.0003850
1: -0.0005447, 0.0006756, -0.0007587, 0.0006212, -0.0006577, 0.0014343
2: 0.0114789, 0.0213211, 0.0119174, 0.0221482, -0.0106693, 0.0053051
3: -0.0044169, -0.0035378, -0.0044759, -0.0035770, -0.0008399, 0.0005372
4: -0.0000936, 0.0041714, -0.0003803, 0.0039813, -0.0040750, 0.0026062
5: -0.0011424, -0.0005058, -0.0011141, -0.0003548, -0.0007876, 0.0003432
6: 0.9911807, 0.9923484, 0.9908193, 0.9922964, -0.0006294, 0.0015292
7: -0.0135523, -0.0058320, -0.0141258, -0.0061759, -0.0041614, 0.0062171
8: -0.0032575, -0.0008388, -0.0042271, -0.0009465, -0.0013037, 0.0033883
9: -0.0056551, -0.0008276, -0.0054400, -0.0003962, -0.0052589, 0.0026021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010407, upper bound: 0.0010592
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010407, upper bound: 0.0010611
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058693, 0.0065133, 0.0058413, 0.0064922, -0.0006229, 0.0003903
1: -0.0005496, 0.0006978, -0.0006546, 0.0006568, -0.0006905, 0.0013524
2: 0.0112994, 0.0213612, 0.0116299, 0.0218622, -0.0105628, 0.0055693
3: -0.0044204, -0.0035218, -0.0044595, -0.0035513, -0.0008692, 0.0005446
4: -0.0001110, 0.0042491, -0.0003005, 0.0041060, -0.0042170, 0.0026422
5: -0.0011540, -0.0005032, -0.0011327, -0.0004333, -0.0007207, 0.0003603
6: 0.9911759, 0.9923696, 0.9910152, 0.9923304, -0.0006608, 0.0013544
7: -0.0135838, -0.0056912, -0.0139479, -0.0059504, -0.0043686, 0.0063559
8: -0.0032674, -0.0007947, -0.0036851, -0.0008759, -0.0013687, 0.0028904
9: -0.0057431, -0.0008079, -0.0055810, -0.0005523, -0.0051908, 0.0027317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010151, upper bound: 0.0010348
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010137, upper bound: 0.0010290
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058272, 0.0064555, 0.0058282, 0.0064911, -0.0006639, 0.0006273
1: -0.0007787, 0.0005858, -0.0007702, 0.0006549, -0.0014336, 0.0013559
2: 0.0122031, 0.0222033, 0.0116459, 0.0221797, -0.0099766, 0.0105574
3: -0.0044791, -0.0036025, -0.0044777, -0.0035527, -0.0009264, 0.0008752
4: -0.0003956, 0.0038575, -0.0003890, 0.0040990, -0.0044946, 0.0042466
5: -0.0010956, -0.0003397, -0.0011316, -0.0003462, -0.0007494, 0.0007919
6: 0.9907815, 0.9922624, 0.9907977, 0.9923286, -0.0015471, 0.0014647
7: -0.0141601, -0.0064000, -0.0141454, -0.0059629, -0.0059467, 0.0059690
8: -0.0043316, -0.0010167, -0.0042868, -0.0008798, -0.0034518, 0.0032701
9: -0.0052999, -0.0003661, -0.0055732, -0.0003790, -0.0049209, 0.0052071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010787, upper bound: 0.0010758
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010752, upper bound: 0.0010757
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058409, 0.0064740, 0.0058256, 0.0065009, -0.0006600, 0.0006484
1: -0.0006581, 0.0006217, -0.0007931, 0.0006737, -0.0013318, 0.0014148
2: 0.0119131, 0.0218718, 0.0114939, 0.0222427, -0.0103296, 0.0103779
3: -0.0044601, -0.0035766, -0.0044814, -0.0035392, -0.0009209, 0.0009048
4: -0.0003032, 0.0039832, -0.0004066, 0.0041649, -0.0044681, 0.0043898
5: -0.0011143, -0.0004307, -0.0011415, -0.0003289, -0.0007854, 0.0007108
6: 0.9910088, 0.9922969, 0.9907544, 0.9923466, -0.0013378, 0.0015424
7: -0.0139538, -0.0061725, -0.0141846, -0.0058437, -0.0061931, 0.0062773
8: -0.0037032, -0.0009455, -0.0044063, -0.0008425, -0.0028608, 0.0034608
9: -0.0054421, -0.0005471, -0.0056477, -0.0003446, -0.0050975, 0.0051007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010969
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010883, upper bound: 0.0010934
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058328, 0.0064755, 0.0058671, 0.0065019, -0.0003806, 0.0006084
1: -0.0007295, 0.0006245, -0.0005538, 0.0006757, -0.0014052, 0.0006785
2: 0.0118904, 0.0220681, 0.0114782, 0.0213946, -0.0054726, 0.0105900
3: -0.0044713, -0.0035746, -0.0044234, -0.0035377, -0.0005310, 0.0008489
4: -0.0003579, 0.0039931, -0.0001255, 0.0041717, -0.0025764, 0.0041186
5: -0.0011158, -0.0003768, -0.0011425, -0.0005010, -0.0003540, 0.0007657
6: 0.9908742, 0.9922996, 0.9911719, 0.9923485, -0.0014743, 0.0006493
7: -0.0140760, -0.0061547, -0.0136100, -0.0058314, -0.0061543, 0.0042928
8: -0.0040754, -0.0009399, -0.0032756, -0.0008386, -0.0032368, 0.0013449
9: -0.0054533, -0.0004399, -0.0056554, -0.0007915, -0.0026843, 0.0052156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010243, upper bound: 0.0009995
time: 1.26 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0009993
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058443, 0.0064949, 0.0058646, 0.0065134, -0.0003866, 0.0006303
1: -0.0006282, 0.0006621, -0.0005588, 0.0006979, -0.0013261, 0.0007136
2: 0.0115878, 0.0217896, 0.0112986, 0.0214351, -0.0057558, 0.0104910
3: -0.0044553, -0.0035475, -0.0044270, -0.0035217, -0.0005394, 0.0008795
4: -0.0002803, 0.0041242, -0.0001430, 0.0042495, -0.0026170, 0.0042672
5: -0.0011354, -0.0004532, -0.0011541, -0.0004984, -0.0003723, 0.0007009
6: 0.9910651, 0.9923354, 0.9911671, 0.9923698, -0.0013047, 0.0006829
7: -0.0139027, -0.0059174, -0.0136418, -0.0056905, -0.0062912, 0.0045149
8: -0.0035474, -0.0008655, -0.0032855, -0.0007945, -0.0027529, 0.0014145
9: -0.0056017, -0.0005919, -0.0057435, -0.0007717, -0.0028232, 0.0051516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010356, upper bound: 0.0010168
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010299, upper bound: 0.0010161
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058178, 0.0064761, 0.0058390, 0.0064727, -0.0006549, 0.0006372
1: -0.0008625, 0.0006258, -0.0006752, 0.0006191, -0.0014816, 0.0013010
2: 0.0118805, 0.0224333, 0.0119340, 0.0219188, -0.0100383, 0.0104994
3: -0.0044923, -0.0035737, -0.0044628, -0.0035785, -0.0009139, 0.0008891
4: -0.0004597, 0.0039973, -0.0003163, 0.0039742, -0.0044339, 0.0043136
5: -0.0011165, -0.0002766, -0.0011130, -0.0004178, -0.0006987, 0.0008364
6: 0.9906238, 0.9923007, 0.9909765, 0.9922944, -0.0016705, 0.0013242
7: -0.0143032, -0.0061470, -0.0139830, -0.0061889, -0.0062206, 0.0059392
8: -0.0047676, -0.0009375, -0.0037922, -0.0009506, -0.0038170, 0.0028548
9: -0.0054581, -0.0002405, -0.0054319, -0.0005214, -0.0049367, 0.0051913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010847, upper bound: 0.0010711
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010821, upper bound: 0.0010710
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058302, 0.0064958, 0.0058363, 0.0064823, -0.0006521, 0.0006595
1: -0.0007524, 0.0006639, -0.0006990, 0.0006378, -0.0013902, 0.0013628
2: 0.0115733, 0.0221310, 0.0117837, 0.0219841, -0.0104108, 0.0103472
3: -0.0044749, -0.0035462, -0.0044665, -0.0035650, -0.0009099, 0.0009203
4: -0.0003754, 0.0041305, -0.0003345, 0.0040393, -0.0044147, 0.0044650
5: -0.0011363, -0.0003596, -0.0011227, -0.0003999, -0.0007365, 0.0007631
6: 0.9908311, 0.9923372, 0.9909318, 0.9923122, -0.0014811, 0.0014054
7: -0.0141150, -0.0059060, -0.0140237, -0.0060711, -0.0062477, 0.0060393
8: -0.0041944, -0.0008620, -0.0039161, -0.0009137, -0.0032807, 0.0030541
9: -0.0056088, -0.0004056, -0.0055056, -0.0004857, -0.0051230, 0.0051000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010988, upper bound: 0.0010898
time: 1.74 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010953, upper bound: 0.0010895
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058513, 0.0064923, 0.0058634, 0.0065010, -0.0003778, 0.0003743
1: -0.0005845, 0.0006571, -0.0005610, 0.0006740, -0.0007318, 0.0007250
2: 0.0116276, 0.0216423, 0.0114915, 0.0214527, -0.0058479, 0.0059025
3: -0.0044455, -0.0035511, -0.0044286, -0.0035389, -0.0005272, 0.0005223
4: -0.0002328, 0.0041069, -0.0001507, 0.0041659, -0.0025578, 0.0025341
5: -0.0011328, -0.0004850, -0.0011416, -0.0004972, -0.0003783, 0.0003818
6: 0.9911426, 0.9923307, 0.9911651, 0.9923469, -0.0007003, 0.0006938
7: -0.0138043, -0.0059486, -0.0136556, -0.0058418, -0.0046300, 0.0045872
8: -0.0033364, -0.0008753, -0.0032899, -0.0008419, -0.0014506, 0.0014371
9: -0.0055822, -0.0006701, -0.0056489, -0.0007630, -0.0028683, 0.0028951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010266, upper bound: 0.0010132
time: 1.04 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010124
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058488, 0.0065021, 0.0058711, 0.0065250, -0.0003944, 0.0003852
1: -0.0005893, 0.0006760, -0.0005461, 0.0007205, -0.0007640, 0.0007460
2: 0.0114751, 0.0216808, 0.0111164, 0.0213331, -0.0060175, 0.0061622
3: -0.0044490, -0.0035375, -0.0044179, -0.0035054, -0.0005504, 0.0005375
4: -0.0002495, 0.0041730, -0.0000988, 0.0043285, -0.0026703, 0.0026076
5: -0.0011427, -0.0004825, -0.0011659, -0.0005050, -0.0003893, 0.0003986
6: 0.9911379, 0.9923488, 0.9911793, 0.9923915, -0.0007311, 0.0007139
7: -0.0138345, -0.0058290, -0.0135617, -0.0055476, -0.0048337, 0.0047202
8: -0.0033459, -0.0008378, -0.0032605, -0.0007497, -0.0015144, 0.0014788
9: -0.0056569, -0.0006511, -0.0058329, -0.0008217, -0.0029515, 0.0030225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010348, upper bound: 0.0010228
time: 1.21 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010290, upper bound: 0.0010211
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058374, 0.0064931, 0.0058296, 0.0064739, -0.0006365, 0.0006635
1: -0.0006887, 0.0006586, -0.0007580, 0.0006214, -0.0013101, 0.0014166
2: 0.0116159, 0.0219558, 0.0119155, 0.0221463, -0.0105304, 0.0100404
3: -0.0044649, -0.0035500, -0.0044758, -0.0035768, -0.0008881, 0.0009258
4: -0.0003266, 0.0041120, -0.0003797, 0.0039822, -0.0043088, 0.0044917
5: -0.0011336, -0.0004076, -0.0011142, -0.0003554, -0.0007782, 0.0007066
6: 0.9909511, 0.9923321, 0.9908205, 0.9922966, -0.0013455, 0.0015116
7: -0.0140061, -0.0059394, -0.0141246, -0.0061744, -0.0060893, 0.0062170
8: -0.0038625, -0.0008724, -0.0042235, -0.0009461, -0.0029164, 0.0033511
9: -0.0055879, -0.0005012, -0.0054409, -0.0003972, -0.0051907, 0.0049398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010758, upper bound: 0.0010837
time: 1.07 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010757, upper bound: 0.0010815
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058349, 0.0065030, 0.0058424, 0.0064923, -0.0006575, 0.0006606
1: -0.0007115, 0.0006779, -0.0006446, 0.0006571, -0.0013686, 0.0013225
2: 0.0114604, 0.0220185, 0.0116276, 0.0218348, -0.0103744, 0.0103909
3: -0.0044685, -0.0035362, -0.0044579, -0.0035511, -0.0009174, 0.0009218
4: -0.0003441, 0.0041794, -0.0002929, 0.0041069, -0.0044510, 0.0044723
5: -0.0011436, -0.0003904, -0.0011328, -0.0004408, -0.0007028, 0.0007424
6: 0.9909081, 0.9923506, 0.9910342, 0.9923307, -0.0014226, 0.0013165
7: -0.0140451, -0.0058174, -0.0139308, -0.0059486, -0.0063093, 0.0064160
8: -0.0039813, -0.0008342, -0.0036330, -0.0008753, -0.0031060, 0.0027988
9: -0.0056642, -0.0004670, -0.0055821, -0.0005673, -0.0050969, 0.0051152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010969, upper bound: 0.0010936
time: 1.02 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010933, upper bound: 0.0010934
time: 1.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.72 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010242, upper bound: 0.0009998
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010193, upper bound: 0.0009996
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010331, upper bound: 0.0010168
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010275, upper bound: 0.0010161
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010751, upper bound: 0.0010760
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010712, upper bound: 0.0010758
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010934, upper bound: 0.0010898
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010894
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010407, upper bound: 0.0010592
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010407, upper bound: 0.0010611
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010151, upper bound: 0.0010348
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010137, upper bound: 0.0010290
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010787, upper bound: 0.0010758
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010752, upper bound: 0.0010757
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010969
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010883, upper bound: 0.0010934
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010243, upper bound: 0.0009995
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0009993
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010356, upper bound: 0.0010168
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010299, upper bound: 0.0010161
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010847, upper bound: 0.0010711
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010821, upper bound: 0.0010710
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010988, upper bound: 0.0010898
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010953, upper bound: 0.0010895
IS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010266, upper bound: 0.0010132
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010124
IS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010348, upper bound: 0.0010228
IS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010290, upper bound: 0.0010211
IS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010758, upper bound: 0.0010837
IS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010757, upper bound: 0.0010815
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010969, upper bound: 0.0010936
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 6, lower bound: -0.0010933, upper bound: 0.0010934

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058375, 0.0064552, 0.0058635, 0.0065020, -0.0003673, 0.0005916
1: -0.0006883, 0.0005852, -0.0005608, 0.0006758, -0.0013641, 0.0006213
2: 0.0122080, 0.0219548, 0.0114766, 0.0214511, -0.0050117, 0.0104782
3: -0.0044648, -0.0036029, -0.0044285, -0.0035376, -0.0005126, 0.0008255
4: -0.0003263, 0.0038554, -0.0001500, 0.0041724, -0.0024869, 0.0040054
5: -0.0010953, -0.0004079, -0.0011426, -0.0004974, -0.0003242, 0.0007347
6: 0.9909518, 0.9922619, 0.9911652, 0.9923487, -0.0013968, 0.0005946
7: -0.0140054, -0.0064038, -0.0136543, -0.0058302, -0.0059229, 0.0039312
8: -0.0038605, -0.0010179, -0.0032895, -0.0008382, -0.0030223, 0.0012316
9: -0.0052975, -0.0005018, -0.0056562, -0.0007638, -0.0024582, 0.0051544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010242, upper bound: 0.0009982
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010242, upper bound: 0.0009998
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058466, 0.0064661, 0.0058707, 0.0065018, -0.0003690, 0.0005954
1: -0.0006078, 0.0006064, -0.0005469, 0.0006756, -0.0012834, 0.0006544
2: 0.0120371, 0.0217337, 0.0114787, 0.0213388, -0.0052780, 0.0102550
3: -0.0044521, -0.0035877, -0.0044184, -0.0035378, -0.0005148, 0.0008308
4: -0.0002647, 0.0039295, -0.0001013, 0.0041715, -0.0024979, 0.0040308
5: -0.0011063, -0.0004686, -0.0011425, -0.0005046, -0.0003414, 0.0006739
6: 0.9911033, 0.9922822, 0.9911786, 0.9923484, -0.0012451, 0.0006262
7: -0.0138679, -0.0062698, -0.0135662, -0.0058318, -0.0059561, 0.0041401
8: -0.0034414, -0.0009759, -0.0032619, -0.0008387, -0.0026027, 0.0012971
9: -0.0053813, -0.0006224, -0.0056552, -0.0008189, -0.0025888, 0.0050328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010193, upper bound: 0.0009980
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010193, upper bound: 0.0009996
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064734, 0.0058610, 0.0065135, -0.0003710, 0.0003420
1: -0.0005892, 0.0006204, -0.0005657, 0.0006981, -0.0007186, 0.0006624
2: 0.0119238, 0.0216800, 0.0112971, 0.0214911, -0.0053429, 0.0057962
3: -0.0044489, -0.0035775, -0.0044320, -0.0035216, -0.0005177, 0.0004772
4: -0.0002492, 0.0039786, -0.0001673, 0.0042502, -0.0025117, 0.0023153
5: -0.0011137, -0.0004825, -0.0011542, -0.0004948, -0.0003456, 0.0003749
6: 0.9911380, 0.9922956, 0.9911605, 0.9923699, -0.0006877, 0.0006339
7: -0.0138339, -0.0061809, -0.0136857, -0.0056893, -0.0045466, 0.0041911
8: -0.0033457, -0.0009481, -0.0032993, -0.0007941, -0.0014244, 0.0013130
9: -0.0054369, -0.0006516, -0.0057443, -0.0007442, -0.0026206, 0.0028430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010331, upper bound: 0.0010121
time: 1.13 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010331, upper bound: 0.0010168
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058581, 0.0064844, 0.0058681, 0.0065133, -0.0003723, 0.0003497
1: -0.0005712, 0.0006418, -0.0005518, 0.0006978, -0.0007210, 0.0006773
2: 0.0117516, 0.0215355, 0.0112992, 0.0213791, -0.0054630, 0.0058156
3: -0.0044360, -0.0035622, -0.0044220, -0.0035218, -0.0005194, 0.0004879
4: -0.0001866, 0.0040532, -0.0001188, 0.0042492, -0.0025201, 0.0023674
5: -0.0011248, -0.0004919, -0.0011541, -0.0005020, -0.0003534, 0.0003762
6: 0.9911552, 0.9923160, 0.9911739, 0.9923697, -0.0006900, 0.0006482
7: -0.0137205, -0.0060458, -0.0135978, -0.0056910, -0.0045619, 0.0042853
8: -0.0033102, -0.0009058, -0.0032718, -0.0007946, -0.0014292, 0.0013426
9: -0.0055213, -0.0007224, -0.0057432, -0.0007992, -0.0026796, 0.0028525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010275, upper bound: 0.0010111
time: 1.09 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010275, upper bound: 0.0010161
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058357, 0.0064729, 0.0058200, 0.0064559, -0.0006202, 0.0006528
1: -0.0007044, 0.0006194, -0.0008424, 0.0005866, -0.0012910, 0.0014618
2: 0.0119317, 0.0219989, 0.0121963, 0.0223781, -0.0104464, 0.0098026
3: -0.0044674, -0.0035782, -0.0044891, -0.0036019, -0.0008655, 0.0009109
4: -0.0003386, 0.0039752, -0.0004443, 0.0038605, -0.0041991, 0.0044195
5: -0.0011131, -0.0003958, -0.0010960, -0.0002918, -0.0008214, 0.0007002
6: 0.9909216, 0.9922947, 0.9906617, 0.9922633, -0.0013418, 0.0016329
7: -0.0140329, -0.0061871, -0.0142688, -0.0063947, -0.0056962, 0.0057949
8: -0.0039442, -0.0009500, -0.0046628, -0.0010151, -0.0029291, 0.0037128
9: -0.0054330, -0.0004777, -0.0053032, -0.0002707, -0.0051623, 0.0048255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010751, upper bound: 0.0010734
time: 1.13 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010751, upper bound: 0.0010760
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058447, 0.0064835, 0.0058275, 0.0064556, -0.0006108, 0.0006560
1: -0.0006243, 0.0006401, -0.0007761, 0.0005860, -0.0012103, 0.0014163
2: 0.0117645, 0.0217790, 0.0122017, 0.0221961, -0.0104315, 0.0095773
3: -0.0044547, -0.0035633, -0.0044787, -0.0036024, -0.0008524, 0.0009154
4: -0.0002773, 0.0040476, -0.0003936, 0.0038582, -0.0041355, 0.0044412
5: -0.0011240, -0.0004562, -0.0010957, -0.0003417, -0.0007822, 0.0006395
6: 0.9910724, 0.9923145, 0.9907865, 0.9922627, -0.0011903, 0.0015280
7: -0.0138960, -0.0060560, -0.0141556, -0.0063989, -0.0057018, 0.0061228
8: -0.0035272, -0.0009090, -0.0043178, -0.0010164, -0.0025108, 0.0034089
9: -0.0055150, -0.0005977, -0.0053006, -0.0003701, -0.0051449, 0.0047028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010712, upper bound: 0.0010733
time: 1.16 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010712, upper bound: 0.0010758
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058330, 0.0064825, 0.0058335, 0.0064745, -0.0006415, 0.0006490
1: -0.0007280, 0.0006381, -0.0007238, 0.0006227, -0.0013506, 0.0013619
2: 0.0117814, 0.0220637, 0.0119056, 0.0220523, -0.0102709, 0.0101582
3: -0.0044711, -0.0035648, -0.0044704, -0.0035759, -0.0008952, 0.0009056
4: -0.0003567, 0.0040403, -0.0003535, 0.0039865, -0.0043432, 0.0043938
5: -0.0011229, -0.0003780, -0.0011148, -0.0003811, -0.0007417, 0.0007368
6: 0.9908771, 0.9923124, 0.9908850, 0.9922978, -0.0014207, 0.0014275
7: -0.0140732, -0.0060692, -0.0140661, -0.0061666, -0.0060386, 0.0060056
8: -0.0040670, -0.0009131, -0.0040454, -0.0009436, -0.0031234, 0.0031323
9: -0.0055067, -0.0004423, -0.0054458, -0.0004485, -0.0050582, 0.0050035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010934, upper bound: 0.0010871
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010934, upper bound: 0.0010898
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058421, 0.0064932, 0.0058411, 0.0064741, -0.0006320, 0.0006521
1: -0.0006476, 0.0006588, -0.0006562, 0.0006219, -0.0012695, 0.0013150
2: 0.0116141, 0.0218430, 0.0119118, 0.0218667, -0.0102525, 0.0099312
3: -0.0044584, -0.0035499, -0.0044598, -0.0035765, -0.0008819, 0.0009099
4: -0.0002952, 0.0041128, -0.0003018, 0.0039838, -0.0042789, 0.0044145
5: -0.0011337, -0.0004386, -0.0011144, -0.0004321, -0.0007016, 0.0006758
6: 0.9910284, 0.9923323, 0.9910122, 0.9922970, -0.0012686, 0.0013201
7: -0.0139359, -0.0059380, -0.0139506, -0.0061715, -0.0060536, 0.0063519
8: -0.0036486, -0.0008720, -0.0036935, -0.0009451, -0.0027034, 0.0028215
9: -0.0055888, -0.0005628, -0.0054428, -0.0005499, -0.0050389, 0.0048800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010887, upper bound: 0.0010867
time: 1.28 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010887, upper bound: 0.0010895
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058719, 0.0065018, 0.0058453, 0.0064734, -0.0006015, 0.0003679
1: -0.0005447, 0.0006756, -0.0006193, 0.0006204, -0.0006569, 0.0012949
2: 0.0114789, 0.0213211, 0.0119238, 0.0217654, -0.0102864, 0.0052982
3: -0.0044169, -0.0035378, -0.0044539, -0.0035776, -0.0008393, 0.0005133
4: -0.0000936, 0.0041714, -0.0002735, 0.0039786, -0.0040722, 0.0024906
5: -0.0011424, -0.0005058, -0.0011137, -0.0004599, -0.0006825, 0.0003427
6: 0.9911807, 0.9923484, 0.9910816, 0.9922956, -0.0006286, 0.0012668
7: -0.0135523, -0.0058320, -0.0138876, -0.0061809, -0.0041560, 0.0059996
8: -0.0032575, -0.0008388, -0.0035014, -0.0009481, -0.0013021, 0.0026626
9: -0.0056551, -0.0008276, -0.0054369, -0.0006052, -0.0050499, 0.0025987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010218
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010176
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058719, 0.0065018, 0.0058496, 0.0064937, -0.0003660, 0.0003758
1: -0.0005447, 0.0006756, -0.0005877, 0.0006599, -0.0007090, 0.0007280
2: 0.0114789, 0.0213211, 0.0116052, 0.0216682, -0.0058716, 0.0057184
3: -0.0044169, -0.0035378, -0.0044479, -0.0035491, -0.0005107, 0.0005244
4: -0.0000936, 0.0041714, -0.0002440, 0.0041166, -0.0024780, 0.0025444
5: -0.0011424, -0.0005058, -0.0011343, -0.0004833, -0.0003798, 0.0003699
6: 0.9911807, 0.9923484, 0.9911395, 0.9923334, -0.0006785, 0.0006966
7: -0.0135523, -0.0058320, -0.0138246, -0.0059311, -0.0044856, 0.0046058
8: -0.0032575, -0.0008388, -0.0033428, -0.0008698, -0.0014053, 0.0014430
9: -0.0056551, -0.0008276, -0.0055931, -0.0006574, -0.0028799, 0.0028048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010193
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010191
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058712, 0.0065133, 0.0058507, 0.0064919, -0.0003463, 0.0003782
1: -0.0005459, 0.0006977, -0.0005857, 0.0006563, -0.0006708, 0.0007326
2: 0.0113001, 0.0213314, 0.0116346, 0.0216521, -0.0059093, 0.0054105
3: -0.0044178, -0.0035218, -0.0044464, -0.0035517, -0.0004832, 0.0005278
4: -0.0000981, 0.0042489, -0.0002371, 0.0041039, -0.0023446, 0.0025607
5: -0.0011540, -0.0005051, -0.0011324, -0.0004844, -0.0003823, 0.0003500
6: 0.9911795, 0.9923695, 0.9911414, 0.9923299, -0.0006419, 0.0007011
7: -0.0135604, -0.0056917, -0.0138120, -0.0059540, -0.0042441, 0.0046353
8: -0.0032600, -0.0007948, -0.0033388, -0.0008770, -0.0013296, 0.0014522
9: -0.0057428, -0.0008225, -0.0055787, -0.0006653, -0.0028984, 0.0026538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010151, upper bound: 0.0010320
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010151, upper bound: 0.0010348
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058780, 0.0065131, 0.0058586, 0.0065048, -0.0003650, 0.0003808
1: -0.0005327, 0.0006975, -0.0005703, 0.0006813, -0.0007069, 0.0007376
2: 0.0113023, 0.0212243, 0.0114328, 0.0215282, -0.0059495, 0.0057019
3: -0.0044082, -0.0035220, -0.0044354, -0.0035337, -0.0005093, 0.0005314
4: -0.0000517, 0.0042479, -0.0001834, 0.0041914, -0.0024709, 0.0025782
5: -0.0011539, -0.0005120, -0.0011454, -0.0004924, -0.0003849, 0.0003688
6: 0.9911922, 0.9923693, 0.9911562, 0.9923539, -0.0006765, 0.0007059
7: -0.0134764, -0.0056934, -0.0137148, -0.0057958, -0.0044727, 0.0046669
8: -0.0032337, -0.0007954, -0.0033084, -0.0008274, -0.0014013, 0.0014621
9: -0.0057417, -0.0008751, -0.0056777, -0.0007260, -0.0029182, 0.0027967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010137, upper bound: 0.0010266
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010137, upper bound: 0.0010290
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058366, 0.0064552, 0.0058301, 0.0064910, -0.0006545, 0.0006251
1: -0.0006964, 0.0005852, -0.0007539, 0.0006546, -0.0013510, 0.0013391
2: 0.0122081, 0.0219771, 0.0116479, 0.0221351, -0.0099270, 0.0103292
3: -0.0044661, -0.0036029, -0.0044752, -0.0035529, -0.0009132, 0.0008722
4: -0.0003326, 0.0038554, -0.0003766, 0.0040982, -0.0044307, 0.0042320
5: -0.0010953, -0.0004018, -0.0011315, -0.0003584, -0.0007368, 0.0007297
6: 0.9909366, 0.9922618, 0.9908283, 0.9923283, -0.0013917, 0.0014336
7: -0.0140193, -0.0064039, -0.0141176, -0.0059645, -0.0057763, 0.0059374
8: -0.0039028, -0.0010180, -0.0042022, -0.0008803, -0.0030226, 0.0031842
9: -0.0052974, -0.0004896, -0.0055722, -0.0004034, -0.0048941, 0.0050826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010787, upper bound: 0.0010736
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010787, upper bound: 0.0010758
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058451, 0.0064661, 0.0058372, 0.0064907, -0.0006456, 0.0006289
1: -0.0006208, 0.0006063, -0.0006909, 0.0006540, -0.0012748, 0.0012972
2: 0.0120376, 0.0217693, 0.0116531, 0.0219620, -0.0099244, 0.0101162
3: -0.0044542, -0.0035877, -0.0044652, -0.0035534, -0.0009008, 0.0008775
4: -0.0002746, 0.0039293, -0.0003283, 0.0040959, -0.0043705, 0.0042576
5: -0.0011063, -0.0004588, -0.0011312, -0.0004059, -0.0007004, 0.0006724
6: 0.9910790, 0.9922820, 0.9909470, 0.9923277, -0.0012487, 0.0013350
7: -0.0138900, -0.0062702, -0.0140099, -0.0059686, -0.0057870, 0.0061246
8: -0.0035089, -0.0009761, -0.0038741, -0.0008816, -0.0026273, 0.0028981
9: -0.0053811, -0.0006030, -0.0055696, -0.0004978, -0.0048832, 0.0049666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010752, upper bound: 0.0010735
time: 1.72 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010752, upper bound: 0.0010757
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058428, 0.0064739, 0.0058348, 0.0065002, -0.0006575, 0.0006391
1: -0.0006415, 0.0006215, -0.0007119, 0.0006724, -0.0013140, 0.0013334
2: 0.0119152, 0.0218264, 0.0115041, 0.0220196, -0.0101044, 0.0103222
3: -0.0044574, -0.0035768, -0.0044685, -0.0035401, -0.0009174, 0.0008918
4: -0.0002905, 0.0039823, -0.0003444, 0.0041604, -0.0044510, 0.0043267
5: -0.0011142, -0.0004432, -0.0011408, -0.0003901, -0.0007241, 0.0006976
6: 0.9910399, 0.9922967, 0.9909073, 0.9923454, -0.0013055, 0.0013894
7: -0.0139255, -0.0061742, -0.0140458, -0.0058517, -0.0061420, 0.0061196
8: -0.0036171, -0.0009460, -0.0039834, -0.0008450, -0.0027721, 0.0030374
9: -0.0054411, -0.0005719, -0.0056427, -0.0004664, -0.0049747, 0.0050708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010948
time: 1.41 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010969
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058501, 0.0064735, 0.0058429, 0.0065131, -0.0006630, 0.0003887
1: -0.0005868, 0.0006207, -0.0006404, 0.0006974, -0.0007993, 0.0012611
2: 0.0119216, 0.0216609, 0.0113026, 0.0218233, -0.0099017, 0.0064471
3: -0.0044472, -0.0035773, -0.0044573, -0.0035221, -0.0009252, 0.0005423
4: -0.0002409, 0.0039795, -0.0002897, 0.0042478, -0.0044887, 0.0026312
5: -0.0011138, -0.0004838, -0.0011538, -0.0004440, -0.0006698, 0.0004171
6: 0.9911403, 0.9922959, 0.9910421, 0.9923694, -0.0007649, 0.0012538
7: -0.0138189, -0.0061792, -0.0139236, -0.0056936, -0.0050572, 0.0061453
8: -0.0033410, -0.0009476, -0.0036112, -0.0007954, -0.0015844, 0.0026636
9: -0.0054380, -0.0006609, -0.0057416, -0.0005736, -0.0048644, 0.0031622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010884, upper bound: 0.0010913
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010884, upper bound: 0.0010933
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058417, 0.0064753, 0.0058690, 0.0065018, -0.0003704, 0.0006063
1: -0.0006511, 0.0006242, -0.0005502, 0.0006756, -0.0013266, 0.0006733
2: 0.0118934, 0.0218525, 0.0114788, 0.0213661, -0.0054306, 0.0103737
3: -0.0044589, -0.0035748, -0.0044209, -0.0035378, -0.0005168, 0.0008460
4: -0.0002978, 0.0039918, -0.0001131, 0.0041714, -0.0025073, 0.0041049
5: -0.0011156, -0.0004360, -0.0011424, -0.0005029, -0.0003513, 0.0007065
6: 0.9910220, 0.9922992, 0.9911754, 0.9923484, -0.0013264, 0.0006443
7: -0.0139418, -0.0061571, -0.0135876, -0.0058319, -0.0059753, 0.0042599
8: -0.0036666, -0.0009406, -0.0032686, -0.0008387, -0.0028278, 0.0013346
9: -0.0054518, -0.0005576, -0.0056551, -0.0008055, -0.0026637, 0.0050975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009995
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009995
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058515, 0.0064857, 0.0058761, 0.0065017, -0.0003731, 0.0003678
1: -0.0005840, 0.0006444, -0.0005364, 0.0006753, -0.0007226, 0.0007123
2: 0.0117306, 0.0216384, 0.0114809, 0.0212543, -0.0057454, 0.0058286
3: -0.0044452, -0.0035603, -0.0044109, -0.0035380, -0.0005206, 0.0005132
4: -0.0002311, 0.0040623, -0.0000647, 0.0041705, -0.0025258, 0.0024897
5: -0.0011262, -0.0004852, -0.0011423, -0.0005101, -0.0003717, 0.0003770
6: 0.9911430, 0.9923186, 0.9911886, 0.9923481, -0.0006915, 0.0006817
7: -0.0138012, -0.0060294, -0.0135000, -0.0058335, -0.0045721, 0.0045068
8: -0.0033355, -0.0009006, -0.0032411, -0.0008393, -0.0014324, 0.0014120
9: -0.0055316, -0.0006720, -0.0056541, -0.0008603, -0.0028181, 0.0028589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0009957
time: 1.19 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0009957
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058535, 0.0064945, 0.0058664, 0.0065133, -0.0003741, 0.0003594
1: -0.0005802, 0.0006613, -0.0005553, 0.0006978, -0.0007246, 0.0006961
2: 0.0115940, 0.0216075, 0.0112993, 0.0214066, -0.0056144, 0.0058445
3: -0.0044424, -0.0035481, -0.0044245, -0.0035218, -0.0005220, 0.0005015
4: -0.0002177, 0.0041215, -0.0001307, 0.0042492, -0.0025326, 0.0024329
5: -0.0011350, -0.0004872, -0.0011541, -0.0005002, -0.0003632, 0.0003781
6: 0.9911467, 0.9923348, 0.9911706, 0.9923697, -0.0006934, 0.0006661
7: -0.0137770, -0.0059222, -0.0136194, -0.0056911, -0.0045845, 0.0044040
8: -0.0033279, -0.0008670, -0.0032785, -0.0007946, -0.0014363, 0.0013798
9: -0.0055986, -0.0006871, -0.0057432, -0.0007857, -0.0027538, 0.0028666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010077
time: 0.98 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010168
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058622, 0.0065057, 0.0058735, 0.0065132, -0.0003766, 0.0003795
1: -0.0005634, 0.0006831, -0.0005414, 0.0006976, -0.0007294, 0.0007352
2: 0.0114182, 0.0214726, 0.0113015, 0.0212951, -0.0059297, 0.0058835
3: -0.0044304, -0.0035324, -0.0044145, -0.0035220, -0.0005255, 0.0005296
4: -0.0001593, 0.0041977, -0.0000824, 0.0042483, -0.0025496, 0.0025696
5: -0.0011464, -0.0004960, -0.0011539, -0.0005074, -0.0003836, 0.0003806
6: 0.9911627, 0.9923555, 0.9911837, 0.9923694, -0.0006980, 0.0007035
7: -0.0136711, -0.0057843, -0.0135320, -0.0056928, -0.0046152, 0.0046513
8: -0.0032947, -0.0008238, -0.0032511, -0.0007952, -0.0014459, 0.0014572
9: -0.0056849, -0.0007533, -0.0057421, -0.0008403, -0.0029084, 0.0028858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010065
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010161
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058266, 0.0064758, 0.0058407, 0.0064726, -0.0006460, 0.0006351
1: -0.0007843, 0.0006251, -0.0006595, 0.0006189, -0.0014032, 0.0012846
2: 0.0118860, 0.0222187, 0.0119360, 0.0218757, -0.0099896, 0.0102827
3: -0.0044800, -0.0035742, -0.0044603, -0.0035786, -0.0009013, 0.0008861
4: -0.0003999, 0.0039949, -0.0003043, 0.0039733, -0.0043732, 0.0042992
5: -0.0011161, -0.0003355, -0.0011129, -0.0004296, -0.0006865, 0.0007774
6: 0.9907711, 0.9923002, 0.9910060, 0.9922941, -0.0015231, 0.0012941
7: -0.0141696, -0.0061513, -0.0139562, -0.0061905, -0.0060605, 0.0060024
8: -0.0043606, -0.0009388, -0.0037106, -0.0009511, -0.0034095, 0.0027717
9: -0.0054554, -0.0003577, -0.0054309, -0.0005449, -0.0049104, 0.0050732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010148
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010700
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058365, 0.0064862, 0.0058486, 0.0064722, -0.0006357, 0.0006376
1: -0.0006973, 0.0006453, -0.0005900, 0.0006181, -0.0013154, 0.0012352
2: 0.0117233, 0.0219795, 0.0119421, 0.0216846, -0.0099614, 0.0100374
3: -0.0044662, -0.0035596, -0.0044493, -0.0035792, -0.0008871, 0.0008897
4: -0.0003332, 0.0040655, -0.0002510, 0.0039707, -0.0043039, 0.0043165
5: -0.0011266, -0.0004011, -0.0011125, -0.0004821, -0.0006446, 0.0007113
6: 0.9909350, 0.9923193, 0.9911369, 0.9922934, -0.0013584, 0.0011824
7: -0.0140208, -0.0060236, -0.0138373, -0.0061953, -0.0060780, 0.0063250
8: -0.0039072, -0.0008988, -0.0033484, -0.0009526, -0.0029547, 0.0024496
9: -0.0055352, -0.0004883, -0.0054279, -0.0006492, -0.0048860, 0.0049396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010148
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010700
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058393, 0.0064951, 0.0058380, 0.0064822, -0.0006429, 0.0006571
1: -0.0006724, 0.0006625, -0.0006833, 0.0006375, -0.0013099, 0.0013458
2: 0.0115842, 0.0219112, 0.0117858, 0.0219411, -0.0103570, 0.0101254
3: -0.0044623, -0.0035472, -0.0044640, -0.0035652, -0.0008971, 0.0009168
4: -0.0003142, 0.0041257, -0.0003225, 0.0040384, -0.0043526, 0.0044483
5: -0.0011356, -0.0004199, -0.0011226, -0.0004117, -0.0007240, 0.0007027
6: 0.9909817, 0.9923359, 0.9909612, 0.9923120, -0.0013303, 0.0013747
7: -0.0139783, -0.0059145, -0.0139969, -0.0060727, -0.0060660, 0.0059905
8: -0.0037779, -0.0008646, -0.0038346, -0.0009142, -0.0028637, 0.0029700
9: -0.0056034, -0.0005256, -0.0055045, -0.0005092, -0.0050942, 0.0049790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010190, upper bound: 0.0010231
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010190, upper bound: 0.0010898
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0065062, 0.0058459, 0.0064818, -0.0006329, 0.0004043
1: -0.0005892, 0.0006840, -0.0006136, 0.0006367, -0.0007518, 0.0012976
2: 0.0114107, 0.0216800, 0.0117922, 0.0217496, -0.0103388, 0.0060639
3: -0.0044489, -0.0035317, -0.0044530, -0.0035658, -0.0008831, 0.0005641
4: -0.0002492, 0.0042009, -0.0002691, 0.0040356, -0.0042848, 0.0027370
5: -0.0011468, -0.0004825, -0.0011222, -0.0004642, -0.0006826, 0.0003923
6: 0.9911381, 0.9923564, 0.9910924, 0.9923112, -0.0007194, 0.0012640
7: -0.0138339, -0.0057785, -0.0138777, -0.0060777, -0.0047567, 0.0063582
8: -0.0033457, -0.0008220, -0.0034715, -0.0009158, -0.0014902, 0.0026495
9: -0.0056885, -0.0006516, -0.0055014, -0.0006138, -0.0050747, 0.0029743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010166, upper bound: 0.0010231
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010166, upper bound: 0.0010894
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058608, 0.0064921, 0.0058653, 0.0065010, -0.0003654, 0.0003715
1: -0.0005661, 0.0006567, -0.0005574, 0.0006739, -0.0007077, 0.0007196
2: 0.0116308, 0.0214936, 0.0114921, 0.0214241, -0.0058046, 0.0057082
3: -0.0044323, -0.0035514, -0.0044261, -0.0035390, -0.0005098, 0.0005184
4: -0.0001684, 0.0041056, -0.0001383, 0.0041657, -0.0024736, 0.0025153
5: -0.0011326, -0.0004946, -0.0011416, -0.0004991, -0.0003755, 0.0003693
6: 0.9911602, 0.9923304, 0.9911684, 0.9923468, -0.0006772, 0.0006887
7: -0.0136877, -0.0059511, -0.0136332, -0.0058423, -0.0044777, 0.0045532
8: -0.0032999, -0.0008761, -0.0032828, -0.0008420, -0.0014028, 0.0014265
9: -0.0055806, -0.0007430, -0.0056486, -0.0007771, -0.0028471, 0.0027998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010266, upper bound: 0.0010088
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010266, upper bound: 0.0010088
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058686, 0.0065029, 0.0058723, 0.0065009, -0.0003679, 0.0003934
1: -0.0005510, 0.0006775, -0.0005438, 0.0006737, -0.0007127, 0.0007620
2: 0.0114630, 0.0213720, 0.0114941, 0.0213144, -0.0061460, 0.0057484
3: -0.0044214, -0.0035364, -0.0044163, -0.0035392, -0.0005134, 0.0005489
4: -0.0001157, 0.0041783, -0.0000907, 0.0041648, -0.0024910, 0.0026633
5: -0.0011435, -0.0005025, -0.0011415, -0.0005062, -0.0003976, 0.0003719
6: 0.9911746, 0.9923502, 0.9911814, 0.9923466, -0.0006820, 0.0007292
7: -0.0135922, -0.0058195, -0.0135471, -0.0058438, -0.0045091, 0.0048210
8: -0.0032700, -0.0008349, -0.0032559, -0.0008425, -0.0014127, 0.0015104
9: -0.0056629, -0.0008026, -0.0056476, -0.0008309, -0.0030145, 0.0028195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010079
time: 1.23 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010079
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058583, 0.0065018, 0.0058730, 0.0065250, -0.0003834, 0.0003774
1: -0.0005709, 0.0006755, -0.0005424, 0.0007204, -0.0007427, 0.0007310
2: 0.0114795, 0.0215324, 0.0111170, 0.0213032, -0.0058960, 0.0059904
3: -0.0044357, -0.0035379, -0.0044153, -0.0035055, -0.0005350, 0.0005266
4: -0.0001852, 0.0041711, -0.0000859, 0.0043282, -0.0025959, 0.0025550
5: -0.0011424, -0.0004921, -0.0011658, -0.0005069, -0.0003814, 0.0003875
6: 0.9911556, 0.9923483, 0.9911829, 0.9923914, -0.0007107, 0.0006995
7: -0.0137181, -0.0058325, -0.0135383, -0.0055481, -0.0046990, 0.0046249
8: -0.0033094, -0.0008389, -0.0032531, -0.0007498, -0.0014722, 0.0014490
9: -0.0056548, -0.0007239, -0.0058326, -0.0008364, -0.0028919, 0.0029382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010034
time: 1.35 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010229
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058661, 0.0065129, 0.0058797, 0.0065249, -0.0003863, 0.0004014
1: -0.0005558, 0.0006970, -0.0005294, 0.0007202, -0.0007483, 0.0007776
2: 0.0113060, 0.0214109, 0.0111191, 0.0211978, -0.0062716, 0.0060354
3: -0.0044249, -0.0035224, -0.0044058, -0.0035057, -0.0005391, 0.0005602
4: -0.0001326, 0.0042463, -0.0000402, 0.0043273, -0.0026154, 0.0027177
5: -0.0011536, -0.0004999, -0.0011657, -0.0005137, -0.0004057, 0.0003904
6: 0.9911700, 0.9923689, 0.9911953, 0.9923910, -0.0007161, 0.0007441
7: -0.0136228, -0.0056963, -0.0134556, -0.0055497, -0.0047343, 0.0049196
8: -0.0032796, -0.0007963, -0.0032272, -0.0007503, -0.0014832, 0.0015413
9: -0.0057399, -0.0007835, -0.0058316, -0.0008881, -0.0030762, 0.0029603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010031
time: 1.20 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010211
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058392, 0.0064929, 0.0058388, 0.0064737, -0.0006345, 0.0006541
1: -0.0006727, 0.0006583, -0.0006764, 0.0006211, -0.0012938, 0.0013347
2: 0.0116181, 0.0219119, 0.0119182, 0.0219221, -0.0103040, 0.0099937
3: -0.0044624, -0.0035502, -0.0044629, -0.0035770, -0.0008853, 0.0009127
4: -0.0003144, 0.0041111, -0.0003172, 0.0039810, -0.0042954, 0.0044283
5: -0.0011334, -0.0004197, -0.0011140, -0.0004169, -0.0007166, 0.0006943
6: 0.9909813, 0.9923319, 0.9909743, 0.9922963, -0.0013151, 0.0013576
7: -0.0139787, -0.0059411, -0.0139851, -0.0061765, -0.0060276, 0.0060412
8: -0.0037792, -0.0008730, -0.0037986, -0.0009467, -0.0028325, 0.0029256
9: -0.0055868, -0.0005252, -0.0054396, -0.0005196, -0.0050672, 0.0049144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010227
time: 1.19 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010837
time: 1.50 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058468, 0.0064926, 0.0058471, 0.0064868, -0.0006400, 0.0006456
1: -0.0006058, 0.0006577, -0.0006036, 0.0006464, -0.0012523, 0.0012613
2: 0.0116228, 0.0217282, 0.0117137, 0.0217221, -0.0100994, 0.0100145
3: -0.0044518, -0.0035507, -0.0044515, -0.0035588, -0.0008930, 0.0009008
4: -0.0002632, 0.0041090, -0.0002615, 0.0040696, -0.0043328, 0.0043705
5: -0.0011331, -0.0004701, -0.0011272, -0.0004718, -0.0006614, 0.0006572
6: 0.9911071, 0.9923313, 0.9911112, 0.9923205, -0.0012134, 0.0012201
7: -0.0138645, -0.0059448, -0.0138607, -0.0060161, -0.0062477, 0.0061791
8: -0.0034311, -0.0008741, -0.0034195, -0.0008965, -0.0025346, 0.0025454
9: -0.0055845, -0.0006254, -0.0055399, -0.0006288, -0.0049558, 0.0049145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010191
time: 1.12 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010815
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058439, 0.0065023, 0.0058443, 0.0064922, -0.0006483, 0.0006580
1: -0.0006316, 0.0006764, -0.0006283, 0.0006569, -0.0012885, 0.0013048
2: 0.0114718, 0.0217990, 0.0116297, 0.0217900, -0.0103182, 0.0101693
3: -0.0044559, -0.0035372, -0.0044554, -0.0035513, -0.0009046, 0.0009182
4: -0.0002829, 0.0041745, -0.0002804, 0.0041060, -0.0043889, 0.0044549
5: -0.0011429, -0.0004507, -0.0011327, -0.0004531, -0.0006898, 0.0006820
6: 0.9910586, 0.9923493, 0.9910648, 0.9923305, -0.0012719, 0.0012845
7: -0.0139085, -0.0058264, -0.0139029, -0.0059502, -0.0061535, 0.0063655
8: -0.0035651, -0.0008370, -0.0035482, -0.0008758, -0.0026893, 0.0027112
9: -0.0056586, -0.0005868, -0.0055811, -0.0005917, -0.0050669, 0.0049943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010290
time: 1.24 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010936
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058528, 0.0065134, 0.0058515, 0.0064920, -0.0003970, 0.0004321
1: -0.0005816, 0.0006979, -0.0005841, 0.0006565, -0.0007690, 0.0008369
2: 0.0112983, 0.0216193, 0.0116325, 0.0216395, -0.0067503, 0.0062026
3: -0.0044435, -0.0035217, -0.0044453, -0.0035515, -0.0005540, 0.0006029
4: -0.0002229, 0.0042496, -0.0002316, 0.0041048, -0.0026878, 0.0029252
5: -0.0011541, -0.0004865, -0.0011325, -0.0004852, -0.0004367, 0.0004012
6: 0.9911453, 0.9923698, 0.9911429, 0.9923302, -0.0007359, 0.0008009
7: -0.0137863, -0.0056903, -0.0138021, -0.0059524, -0.0048655, 0.0052950
8: -0.0033308, -0.0007944, -0.0033358, -0.0008765, -0.0015243, 0.0016589
9: -0.0057436, -0.0006813, -0.0055798, -0.0006714, -0.0033109, 0.0030423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010154, upper bound: 0.0010289
time: 1.27 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010154, upper bound: 0.0010933
time: 1.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.99 seconds
IS_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010242, upper bound: 0.0009982
IS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010242, upper bound: 0.0009998
IS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010193, upper bound: 0.0009980
IS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010193, upper bound: 0.0009996
IS_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010331, upper bound: 0.0010121
IS_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010331, upper bound: 0.0010168
IS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010275, upper bound: 0.0010111
IS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010275, upper bound: 0.0010161
IS_A1_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010751, upper bound: 0.0010734
IS_A1_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010751, upper bound: 0.0010760
IS_A1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010712, upper bound: 0.0010733
IS_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010712, upper bound: 0.0010758
IS_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010934, upper bound: 0.0010871
IS_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010934, upper bound: 0.0010898
IS_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010887, upper bound: 0.0010867
IS_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010887, upper bound: 0.0010895
IS_A1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010218
IS_A1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010176
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010193
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010191
IS_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010151, upper bound: 0.0010320
IS_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010151, upper bound: 0.0010348
IS_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010137, upper bound: 0.0010266
IS_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010137, upper bound: 0.0010290
IS_A1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010787, upper bound: 0.0010736
IS_A1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010787, upper bound: 0.0010758
IS_A1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010752, upper bound: 0.0010735
IS_A1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010752, upper bound: 0.0010757
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010948
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010886, upper bound: 0.0010969
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010884, upper bound: 0.0010913
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010884, upper bound: 0.0010933
IS_A2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009995
IS_A2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009995
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0009957
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0009957
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010077
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010168
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010065
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010161
IS_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010148
IS_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010700
IS_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010148
IS_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010700
IS_A2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010190, upper bound: 0.0010231
IS_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010190, upper bound: 0.0010898
IS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010166, upper bound: 0.0010231
IS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010166, upper bound: 0.0010894
IS_A2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010266, upper bound: 0.0010088
IS_A2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010266, upper bound: 0.0010088
IS_A2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010079
IS_A2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010079
IS_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010034
IS_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010229
IS_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010031
IS_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010211
IS_A2_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010227
IS_A2_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010837
IS_A2_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010191
IS_A2_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010815
IS_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010290
IS_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010936
IS_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010154, upper bound: 0.0010289
IS_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 6, lower bound: -0.0010154, upper bound: 0.0010933

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058375, 0.0064552, 0.0058794, 0.0065016, -0.0003669, 0.0005758
1: -0.0006883, 0.0005852, -0.0005301, 0.0006751, -0.0013634, 0.0005855
2: 0.0122080, 0.0219548, 0.0114827, 0.0212036, -0.0047227, 0.0104721
3: -0.0044648, -0.0036029, -0.0044064, -0.0035381, -0.0005120, 0.0008034
4: -0.0003263, 0.0038554, -0.0000427, 0.0041697, -0.0024842, 0.0038982
5: -0.0010953, -0.0004079, -0.0011422, -0.0005134, -0.0003055, 0.0007343
6: 0.9909518, 0.9922619, 0.9911945, 0.9923480, -0.0013961, 0.0005603
7: -0.0140054, -0.0064038, -0.0134602, -0.0058349, -0.0058943, 0.0037046
8: -0.0038605, -0.0010179, -0.0032286, -0.0008397, -0.0030208, 0.0011606
9: -0.0052975, -0.0005018, -0.0056532, -0.0008852, -0.0023165, 0.0051515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009982
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009982
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058375, 0.0064552, 0.0058824, 0.0065253, -0.0004101, 0.0005728
1: -0.0006883, 0.0005852, -0.0005243, 0.0007210, -0.0014093, 0.0006097
2: 0.0122080, 0.0219548, 0.0111124, 0.0211570, -0.0049180, 0.0108424
3: -0.0044648, -0.0036029, -0.0044022, -0.0035051, -0.0005722, 0.0007993
4: -0.0003263, 0.0038554, -0.0000225, 0.0043302, -0.0027764, 0.0038780
5: -0.0010953, -0.0004079, -0.0011661, -0.0005164, -0.0003181, 0.0007582
6: 0.9909518, 0.9922619, 0.9912001, 0.9923919, -0.0014400, 0.0005835
7: -0.0140054, -0.0064038, -0.0134237, -0.0055444, -0.0062760, 0.0038578
8: -0.0038605, -0.0010179, -0.0032172, -0.0007487, -0.0031118, 0.0012086
9: -0.0052975, -0.0005018, -0.0058349, -0.0009081, -0.0024122, 0.0053331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009998
time: 1.37 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009998
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058466, 0.0064661, 0.0058868, 0.0065015, -0.0003686, 0.0005793
1: -0.0006078, 0.0006064, -0.0005158, 0.0006748, -0.0012827, 0.0006186
2: 0.0120371, 0.0217337, 0.0114847, 0.0210880, -0.0049899, 0.0102490
3: -0.0044521, -0.0035877, -0.0043960, -0.0035383, -0.0005143, 0.0008084
4: -0.0002647, 0.0039295, 0.0000074, 0.0041688, -0.0024952, 0.0039221
5: -0.0011063, -0.0004686, -0.0011421, -0.0005208, -0.0003228, 0.0006735
6: 0.9911033, 0.9922822, 0.9912084, 0.9923477, -0.0012444, 0.0005920
7: -0.0138679, -0.0062698, -0.0133695, -0.0058365, -0.0059275, 0.0039142
8: -0.0034414, -0.0009759, -0.0032002, -0.0008402, -0.0026013, 0.0012263
9: -0.0053813, -0.0006224, -0.0056522, -0.0009419, -0.0024475, 0.0050298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009980
time: 1.11 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009980
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058466, 0.0064661, 0.0058893, 0.0065252, -0.0004117, 0.0005768
1: -0.0006078, 0.0006064, -0.0005109, 0.0007207, -0.0013286, 0.0006416
2: 0.0120371, 0.0217337, 0.0111146, 0.0210484, -0.0051753, 0.0106191
3: -0.0044521, -0.0035877, -0.0043925, -0.0035053, -0.0005745, 0.0008048
4: -0.0002647, 0.0039295, 0.0000245, 0.0043292, -0.0027873, 0.0039050
5: -0.0011063, -0.0004686, -0.0011660, -0.0005234, -0.0003348, 0.0006974
6: 0.9911033, 0.9922822, 0.9912130, 0.9923916, -0.0012883, 0.0006140
7: -0.0138679, -0.0062698, -0.0133384, -0.0055462, -0.0063078, 0.0040596
8: -0.0034414, -0.0009759, -0.0031905, -0.0007492, -0.0026922, 0.0012718
9: -0.0053813, -0.0006224, -0.0058338, -0.0009614, -0.0025384, 0.0052113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009996
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009996
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064734, 0.0058768, 0.0065131, -0.0003706, 0.0003239
1: -0.0005892, 0.0006204, -0.0005351, 0.0006974, -0.0007178, 0.0006274
2: 0.0119238, 0.0216800, 0.0113031, 0.0212436, -0.0050608, 0.0057898
3: -0.0044489, -0.0035775, -0.0044099, -0.0035221, -0.0005171, 0.0004520
4: -0.0002492, 0.0039786, -0.0000600, 0.0042475, -0.0025089, 0.0021930
5: -0.0011137, -0.0004825, -0.0011538, -0.0005108, -0.0003274, 0.0003745
6: 0.9911380, 0.9922956, 0.9911899, 0.9923692, -0.0006869, 0.0006004
7: -0.0138339, -0.0061809, -0.0134915, -0.0056941, -0.0045416, 0.0039697
8: -0.0033457, -0.0009481, -0.0032385, -0.0007956, -0.0014229, 0.0012437
9: -0.0054369, -0.0006516, -0.0057413, -0.0008656, -0.0024822, 0.0028398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010069
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010121
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064734, 0.0058798, 0.0065363, -0.0004124, 0.0003359
1: -0.0005892, 0.0006204, -0.0005292, 0.0007422, -0.0007987, 0.0006506
2: 0.0119238, 0.0216800, 0.0109411, 0.0211966, -0.0052475, 0.0064425
3: -0.0044489, -0.0035775, -0.0044057, -0.0034898, -0.0005754, 0.0004687
4: -0.0002492, 0.0039786, -0.0000397, 0.0044044, -0.0027918, 0.0022739
5: -0.0011137, -0.0004825, -0.0011772, -0.0005138, -0.0003395, 0.0004168
6: 0.9911380, 0.9922956, 0.9911954, 0.9924121, -0.0007644, 0.0006226
7: -0.0138339, -0.0061809, -0.0134547, -0.0054101, -0.0050536, 0.0041162
8: -0.0033457, -0.0009481, -0.0032269, -0.0007066, -0.0015833, 0.0012896
9: -0.0054369, -0.0006516, -0.0059189, -0.0008887, -0.0025738, 0.0031600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010081
time: 1.06 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010168
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058581, 0.0064844, 0.0058842, 0.0065130, -0.0003718, 0.0003308
1: -0.0005712, 0.0006418, -0.0005207, 0.0006971, -0.0007202, 0.0006406
2: 0.0117516, 0.0215355, 0.0113053, 0.0211279, -0.0051674, 0.0058092
3: -0.0044360, -0.0035622, -0.0043996, -0.0035223, -0.0005189, 0.0004615
4: -0.0001866, 0.0040532, -0.0000099, 0.0042466, -0.0025174, 0.0022392
5: -0.0011248, -0.0004919, -0.0011537, -0.0005183, -0.0003343, 0.0003758
6: 0.9911552, 0.9923160, 0.9912035, 0.9923689, -0.0006892, 0.0006131
7: -0.0137205, -0.0060458, -0.0134008, -0.0056957, -0.0045569, 0.0040534
8: -0.0033102, -0.0009058, -0.0032100, -0.0007961, -0.0014276, 0.0012699
9: -0.0055213, -0.0007224, -0.0057403, -0.0009224, -0.0025346, 0.0028494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010055
time: 1.13 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010111
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058581, 0.0064844, 0.0058868, 0.0065361, -0.0004136, 0.0003442
1: -0.0005712, 0.0006418, -0.0005157, 0.0007420, -0.0008011, 0.0006667
2: 0.0117516, 0.0215355, 0.0109434, 0.0210878, -0.0053778, 0.0064617
3: -0.0044360, -0.0035622, -0.0043960, -0.0034900, -0.0005771, 0.0004803
4: -0.0001866, 0.0040532, 0.0000075, 0.0044034, -0.0028001, 0.0023304
5: -0.0011248, -0.0004919, -0.0011771, -0.0005209, -0.0003479, 0.0004180
6: 0.9911552, 0.9923160, 0.9912083, 0.9924119, -0.0007666, 0.0006380
7: -0.0137205, -0.0060458, -0.0133693, -0.0054119, -0.0050687, 0.0042184
8: -0.0033102, -0.0009058, -0.0032002, -0.0007072, -0.0015880, 0.0013216
9: -0.0055213, -0.0007224, -0.0059177, -0.0009420, -0.0026377, 0.0031694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010066
time: 1.08 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010161
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058357, 0.0064729, 0.0058354, 0.0064552, -0.0006196, 0.0006375
1: -0.0007044, 0.0006194, -0.0007068, 0.0005853, -0.0012896, 0.0013262
2: 0.0119317, 0.0219989, 0.0122073, 0.0220057, -0.0100740, 0.0097917
3: -0.0044674, -0.0035782, -0.0044677, -0.0036029, -0.0008645, 0.0008895
4: -0.0003386, 0.0039752, -0.0003405, 0.0038557, -0.0041944, 0.0043157
5: -0.0011131, -0.0003958, -0.0010953, -0.0003940, -0.0007192, 0.0006995
6: 0.9909216, 0.9922947, 0.9909170, 0.9922620, -0.0013404, 0.0013777
7: -0.0140329, -0.0061871, -0.0140371, -0.0064033, -0.0056765, 0.0055682
8: -0.0039442, -0.0009500, -0.0039569, -0.0010178, -0.0029264, 0.0030069
9: -0.0054330, -0.0004777, -0.0052978, -0.0004740, -0.0049590, 0.0048202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010128
time: 1.13 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010734
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058357, 0.0064729, 0.0058413, 0.0064752, -0.0006396, 0.0006316
1: -0.0007044, 0.0006194, -0.0006547, 0.0006240, -0.0013284, 0.0012742
2: 0.0119317, 0.0219989, 0.0118945, 0.0218626, -0.0099309, 0.0101044
3: -0.0044674, -0.0035782, -0.0044595, -0.0035749, -0.0008924, 0.0008813
4: -0.0003386, 0.0039752, -0.0003006, 0.0039913, -0.0043299, 0.0042758
5: -0.0011131, -0.0003958, -0.0011155, -0.0004332, -0.0006799, 0.0007198
6: 0.9909216, 0.9922947, 0.9910150, 0.9922991, -0.0013776, 0.0012797
7: -0.0140329, -0.0061871, -0.0139481, -0.0061580, -0.0059301, 0.0056685
8: -0.0039442, -0.0009500, -0.0036857, -0.0009409, -0.0030033, 0.0027357
9: -0.0054330, -0.0004777, -0.0054512, -0.0005521, -0.0048809, 0.0049736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010138
time: 1.09 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010760
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058447, 0.0064835, 0.0058429, 0.0064550, -0.0006103, 0.0006407
1: -0.0006243, 0.0006401, -0.0006406, 0.0005848, -0.0012091, 0.0012808
2: 0.0117645, 0.0217790, 0.0122111, 0.0218238, -0.0100592, 0.0095679
3: -0.0044547, -0.0035633, -0.0044573, -0.0036032, -0.0008515, 0.0008940
4: -0.0002773, 0.0040476, -0.0002898, 0.0038541, -0.0041314, 0.0043374
5: -0.0011240, -0.0004562, -0.0010951, -0.0004439, -0.0006801, 0.0006389
6: 0.9910724, 0.9923145, 0.9910417, 0.9922616, -0.0011892, 0.0012729
7: -0.0138960, -0.0060560, -0.0139239, -0.0064063, -0.0056717, 0.0058961
8: -0.0035272, -0.0009090, -0.0036122, -0.0010187, -0.0025085, 0.0027032
9: -0.0055150, -0.0005977, -0.0052960, -0.0005733, -0.0049417, 0.0046982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010127
time: 1.13 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010733
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058447, 0.0064835, 0.0058491, 0.0064751, -0.0003780, 0.0006345
1: -0.0006243, 0.0006401, -0.0005888, 0.0006238, -0.0012481, 0.0007409
2: 0.0117645, 0.0217790, 0.0118965, 0.0216772, -0.0059761, 0.0098825
3: -0.0044547, -0.0035633, -0.0044487, -0.0035751, -0.0005274, 0.0008853
4: -0.0002773, 0.0040476, -0.0002479, 0.0039904, -0.0025588, 0.0042955
5: -0.0011240, -0.0004562, -0.0011154, -0.0004827, -0.0003866, 0.0006593
6: 0.9910724, 0.9923145, 0.9911383, 0.9922988, -0.0012265, 0.0007090
7: -0.0138960, -0.0060560, -0.0138317, -0.0061595, -0.0059404, 0.0046878
8: -0.0035272, -0.0009090, -0.0033450, -0.0009414, -0.0025858, 0.0014686
9: -0.0055150, -0.0005977, -0.0054503, -0.0006529, -0.0029312, 0.0048525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010136
time: 1.20 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010758
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058330, 0.0064825, 0.0058489, 0.0064738, -0.0003817, 0.0006335
1: -0.0007280, 0.0006381, -0.0005890, 0.0006213, -0.0013492, 0.0006776
2: 0.0117814, 0.0220637, 0.0119167, 0.0216790, -0.0054658, 0.0101471
3: -0.0044711, -0.0035648, -0.0044488, -0.0035769, -0.0005325, 0.0008840
4: -0.0003567, 0.0040403, -0.0002488, 0.0039817, -0.0025838, 0.0042891
5: -0.0011229, -0.0003780, -0.0011141, -0.0004826, -0.0003536, 0.0007361
6: 0.9908771, 0.9923124, 0.9911383, 0.9922965, -0.0014194, 0.0006485
7: -0.0140732, -0.0060692, -0.0138331, -0.0061753, -0.0060183, 0.0042875
8: -0.0040670, -0.0009131, -0.0033455, -0.0009463, -0.0031207, 0.0013432
9: -0.0055067, -0.0004423, -0.0054404, -0.0006520, -0.0026809, 0.0049981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010214
time: 1.17 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010869
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058330, 0.0064825, 0.0058541, 0.0064945, -0.0004056, 0.0006284
1: -0.0007280, 0.0006381, -0.0005790, 0.0006614, -0.0013893, 0.0006981
2: 0.0117814, 0.0220637, 0.0115934, 0.0215980, -0.0056305, 0.0104704
3: -0.0044711, -0.0035648, -0.0044416, -0.0035480, -0.0005659, 0.0008768
4: -0.0003567, 0.0040403, -0.0002136, 0.0041218, -0.0027457, 0.0042540
5: -0.0011229, -0.0003780, -0.0011350, -0.0004878, -0.0003642, 0.0007570
6: 0.9908771, 0.9923124, 0.9911478, 0.9923348, -0.0014577, 0.0006680
7: -0.0140732, -0.0060692, -0.0137696, -0.0059217, -0.0062596, 0.0044166
8: -0.0040670, -0.0009131, -0.0033256, -0.0008669, -0.0032001, 0.0013837
9: -0.0055067, -0.0004423, -0.0055989, -0.0006918, -0.0027617, 0.0051567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010231
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010898
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058421, 0.0064932, 0.0058567, 0.0064734, -0.0003822, 0.0006365
1: -0.0006476, 0.0006588, -0.0005741, 0.0006205, -0.0012681, 0.0007386
2: 0.0116141, 0.0218430, 0.0119230, 0.0215582, -0.0059578, 0.0099200
3: -0.0044584, -0.0035499, -0.0044380, -0.0035775, -0.0005333, 0.0008882
4: -0.0002952, 0.0041128, -0.0001964, 0.0039789, -0.0025877, 0.0043092
5: -0.0011337, -0.0004386, -0.0011137, -0.0004904, -0.0003854, 0.0006751
6: 0.9910284, 0.9923323, 0.9911525, 0.9922957, -0.0012673, 0.0007069
7: -0.0139359, -0.0059380, -0.0137384, -0.0061803, -0.0060334, 0.0046734
8: -0.0036486, -0.0008720, -0.0033158, -0.0009479, -0.0027007, 0.0014642
9: -0.0055888, -0.0005628, -0.0054373, -0.0007113, -0.0029222, 0.0048745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010213
time: 1.20 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010865
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058421, 0.0064932, 0.0058620, 0.0064943, -0.0004004, 0.0006312
1: -0.0006476, 0.0006588, -0.0005637, 0.0006609, -0.0013085, 0.0007576
2: 0.0116141, 0.0218430, 0.0115969, 0.0214750, -0.0061110, 0.0102461
3: -0.0044584, -0.0035499, -0.0044306, -0.0035483, -0.0005587, 0.0008807
4: -0.0002952, 0.0041128, -0.0001603, 0.0041202, -0.0027106, 0.0042731
5: -0.0011337, -0.0004386, -0.0011348, -0.0004958, -0.0003953, 0.0006962
6: 0.9910284, 0.9923323, 0.9911624, 0.9923344, -0.0013060, 0.0007250
7: -0.0139359, -0.0059380, -0.0136731, -0.0059245, -0.0062787, 0.0047935
8: -0.0036486, -0.0008720, -0.0032953, -0.0008678, -0.0027808, 0.0015018
9: -0.0055888, -0.0005628, -0.0055972, -0.0007521, -0.0029974, 0.0050344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010231
time: 1.17 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010894
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058738, 0.0065018, 0.0058544, 0.0064732, -0.0003364, 0.0003582
1: -0.0005410, 0.0006755, -0.0005785, 0.0006201, -0.0006516, 0.0006938
2: 0.0114796, 0.0212912, 0.0119266, 0.0215942, -0.0055958, 0.0052561
3: -0.0044142, -0.0035379, -0.0044413, -0.0035778, -0.0004694, 0.0004998
4: -0.0000807, 0.0041711, -0.0002120, 0.0039774, -0.0022777, 0.0024249
5: -0.0011424, -0.0005077, -0.0011135, -0.0004881, -0.0003620, 0.0003400
6: 0.9911842, 0.9923483, 0.9911482, 0.9922953, -0.0006236, 0.0006639
7: -0.0135289, -0.0058325, -0.0137666, -0.0061831, -0.0041230, 0.0043895
8: -0.0032502, -0.0008389, -0.0033246, -0.0009488, -0.0012917, 0.0013752
9: -0.0056547, -0.0008423, -0.0054355, -0.0006936, -0.0027447, 0.0025781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008508, upper bound: 0.0008727
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008236, upper bound: 0.0008540
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058806, 0.0065017, 0.0058627, 0.0064863, -0.0003549, 0.0003611
1: -0.0005278, 0.0006752, -0.0005623, 0.0006454, -0.0006875, 0.0006995
2: 0.0114817, 0.0211850, 0.0117219, 0.0214636, -0.0056421, 0.0055450
3: -0.0044047, -0.0035381, -0.0044296, -0.0035595, -0.0004953, 0.0005039
4: -0.0000347, 0.0041702, -0.0001554, 0.0040661, -0.0024029, 0.0024450
5: -0.0011423, -0.0005146, -0.0011267, -0.0004965, -0.0003650, 0.0003587
6: 0.9911968, 0.9923481, 0.9911637, 0.9923195, -0.0006579, 0.0006694
7: -0.0134456, -0.0058341, -0.0136641, -0.0060226, -0.0043496, 0.0044258
8: -0.0032241, -0.0008394, -0.0032925, -0.0008985, -0.0013627, 0.0013866
9: -0.0056537, -0.0008943, -0.0055359, -0.0007577, -0.0027674, 0.0027198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009243, upper bound: 0.0009729
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009256, upper bound: 0.0009449
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058815, 0.0065016, 0.0058515, 0.0064937, -0.0003538, 0.0003733
1: -0.0005261, 0.0006751, -0.0005841, 0.0006598, -0.0006854, 0.0007230
2: 0.0114822, 0.0211710, 0.0116058, 0.0216392, -0.0058315, 0.0055280
3: -0.0044035, -0.0035381, -0.0044453, -0.0035491, -0.0004937, 0.0005208
4: -0.0000286, 0.0041699, -0.0002315, 0.0041164, -0.0023955, 0.0025270
5: -0.0011422, -0.0005155, -0.0011342, -0.0004852, -0.0003772, 0.0003576
6: 0.9911985, 0.9923480, 0.9911429, 0.9923333, -0.0006559, 0.0006919
7: -0.0134346, -0.0058346, -0.0138019, -0.0059315, -0.0043363, 0.0045744
8: -0.0032206, -0.0008396, -0.0033357, -0.0008700, -0.0013585, 0.0014331
9: -0.0056535, -0.0009012, -0.0055928, -0.0006715, -0.0028603, 0.0027114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008533, upper bound: 0.0008619
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008197, upper bound: 0.0008339
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058886, 0.0065139, 0.0058587, 0.0064936, -0.0003554, 0.0004034
1: -0.0005122, 0.0006989, -0.0005701, 0.0006596, -0.0006883, 0.0007813
2: 0.0112908, 0.0210595, 0.0116077, 0.0215265, -0.0063021, 0.0055521
3: -0.0043935, -0.0035210, -0.0044352, -0.0035493, -0.0004959, 0.0005629
4: 0.0000197, 0.0042529, -0.0001826, 0.0041155, -0.0024059, 0.0027309
5: -0.0011546, -0.0005227, -0.0011341, -0.0004925, -0.0004077, 0.0003592
6: 0.9912117, 0.9923707, 0.9911563, 0.9923331, -0.0006587, 0.0007477
7: -0.0133471, -0.0056844, -0.0137134, -0.0059330, -0.0043551, 0.0049435
8: -0.0031932, -0.0007925, -0.0033080, -0.0008704, -0.0013644, 0.0015488
9: -0.0057473, -0.0009559, -0.0055919, -0.0007269, -0.0030911, 0.0027232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008427, upper bound: 0.0008569
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008114, upper bound: 0.0008304
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058712, 0.0065133, 0.0058665, 0.0064915, -0.0003459, 0.0003613
1: -0.0005459, 0.0006977, -0.0005550, 0.0006555, -0.0006700, 0.0006998
2: 0.0113001, 0.0213314, 0.0116406, 0.0214041, -0.0056445, 0.0054042
3: -0.0044178, -0.0035218, -0.0044243, -0.0035522, -0.0004827, 0.0005041
4: -0.0000981, 0.0042489, -0.0001296, 0.0041013, -0.0023418, 0.0024460
5: -0.0011540, -0.0005051, -0.0011320, -0.0005004, -0.0003651, 0.0003496
6: 0.9911795, 0.9923695, 0.9911708, 0.9923291, -0.0006412, 0.0006697
7: -0.0135604, -0.0056917, -0.0136175, -0.0059588, -0.0042391, 0.0044277
8: -0.0032600, -0.0007948, -0.0032779, -0.0008785, -0.0013281, 0.0013872
9: -0.0057428, -0.0008225, -0.0055758, -0.0007869, -0.0027686, 0.0026507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010258
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010318
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058712, 0.0065133, 0.0058709, 0.0065130, -0.0003724, 0.0003702
1: -0.0005459, 0.0006977, -0.0005465, 0.0006972, -0.0007213, 0.0007171
2: 0.0113001, 0.0213314, 0.0113045, 0.0213358, -0.0057839, 0.0058180
3: -0.0044178, -0.0035218, -0.0044182, -0.0035222, -0.0005196, 0.0005166
4: -0.0000981, 0.0042489, -0.0001000, 0.0042469, -0.0025212, 0.0025064
5: -0.0011540, -0.0005051, -0.0011537, -0.0005048, -0.0003742, 0.0003764
6: 0.9911795, 0.9923695, 0.9911789, 0.9923691, -0.0006903, 0.0006862
7: -0.0135604, -0.0056917, -0.0135639, -0.0056951, -0.0045638, 0.0045370
8: -0.0032600, -0.0007948, -0.0032611, -0.0007959, -0.0014298, 0.0014214
9: -0.0057428, -0.0008225, -0.0057406, -0.0008204, -0.0028370, 0.0028537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010267
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010344
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058780, 0.0065131, 0.0058745, 0.0065044, -0.0003646, 0.0003637
1: -0.0005327, 0.0006975, -0.0005395, 0.0006805, -0.0007061, 0.0007045
2: 0.0113023, 0.0212243, 0.0114388, 0.0212795, -0.0056821, 0.0056956
3: -0.0044082, -0.0035220, -0.0044131, -0.0035342, -0.0005087, 0.0005075
4: -0.0000517, 0.0042479, -0.0000756, 0.0041887, -0.0024681, 0.0024623
5: -0.0011539, -0.0005120, -0.0011450, -0.0005085, -0.0003676, 0.0003684
6: 0.9911922, 0.9923693, 0.9911856, 0.9923531, -0.0006757, 0.0006741
7: -0.0134764, -0.0056934, -0.0135197, -0.0058005, -0.0044677, 0.0044571
8: -0.0032337, -0.0007954, -0.0032473, -0.0008289, -0.0013997, 0.0013964
9: -0.0057417, -0.0008751, -0.0056748, -0.0008480, -0.0027870, 0.0027936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010195
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010264
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058780, 0.0065131, 0.0058779, 0.0065258, -0.0003919, 0.0003726
1: -0.0005327, 0.0006975, -0.0005329, 0.0007220, -0.0007590, 0.0007216
2: 0.0113023, 0.0212243, 0.0111046, 0.0212259, -0.0058204, 0.0061220
3: -0.0044082, -0.0035220, -0.0044084, -0.0035044, -0.0005468, 0.0005199
4: -0.0000517, 0.0042479, -0.0000524, 0.0043336, -0.0026529, 0.0025222
5: -0.0011539, -0.0005120, -0.0011666, -0.0005119, -0.0003765, 0.0003960
6: 0.9911922, 0.9923693, 0.9911920, 0.9923928, -0.0007263, 0.0006906
7: -0.0134764, -0.0056934, -0.0134777, -0.0055383, -0.0048022, 0.0045656
8: -0.0032337, -0.0007954, -0.0032341, -0.0007468, -0.0015045, 0.0014304
9: -0.0057417, -0.0008751, -0.0058387, -0.0008743, -0.0028549, 0.0030028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010208
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010289
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058366, 0.0064552, 0.0058457, 0.0064903, -0.0006537, 0.0006095
1: -0.0006964, 0.0005852, -0.0006158, 0.0006532, -0.0013497, 0.0012009
2: 0.0122081, 0.0219771, 0.0116590, 0.0217556, -0.0095474, 0.0103181
3: -0.0044661, -0.0036029, -0.0044534, -0.0035539, -0.0009122, 0.0008504
4: -0.0003326, 0.0038554, -0.0002708, 0.0040933, -0.0044259, 0.0041262
5: -0.0010953, -0.0004018, -0.0011308, -0.0004626, -0.0006327, 0.0007290
6: 0.9909366, 0.9922618, 0.9910884, 0.9923271, -0.0013905, 0.0011734
7: -0.0140193, -0.0064039, -0.0138815, -0.0059732, -0.0057571, 0.0057151
8: -0.0039028, -0.0010180, -0.0034829, -0.0008830, -0.0030198, 0.0024649
9: -0.0052974, -0.0004896, -0.0055667, -0.0006105, -0.0046869, 0.0050772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010011
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010716
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058366, 0.0064552, 0.0058502, 0.0065115, -0.0003991, 0.0006050
1: -0.0006964, 0.0005852, -0.0005866, 0.0006943, -0.0013907, 0.0006880
2: 0.0122081, 0.0219771, 0.0113281, 0.0216596, -0.0055495, 0.0106490
3: -0.0044661, -0.0036029, -0.0044471, -0.0035243, -0.0005568, 0.0008442
4: -0.0003326, 0.0038554, -0.0002403, 0.0042367, -0.0027016, 0.0040957
5: -0.0010953, -0.0004018, -0.0011522, -0.0004839, -0.0003590, 0.0007504
6: 0.9909366, 0.9922618, 0.9911405, 0.9923663, -0.0014297, 0.0006584
7: -0.0140193, -0.0064039, -0.0138178, -0.0057136, -0.0062647, 0.0043531
8: -0.0039028, -0.0010180, -0.0033407, -0.0008017, -0.0031011, 0.0013638
9: -0.0052974, -0.0004896, -0.0057291, -0.0006616, -0.0027220, 0.0052395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010035
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010744
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058451, 0.0064661, 0.0058529, 0.0064901, -0.0003643, 0.0006132
1: -0.0006208, 0.0006063, -0.0005813, 0.0006529, -0.0012737, 0.0007077
2: 0.0120376, 0.0217693, 0.0116620, 0.0216168, -0.0057083, 0.0101073
3: -0.0044542, -0.0035877, -0.0044433, -0.0035542, -0.0005084, 0.0008556
4: -0.0002746, 0.0039293, -0.0002218, 0.0040920, -0.0024666, 0.0041511
5: -0.0011063, -0.0004588, -0.0011306, -0.0004866, -0.0003693, 0.0006718
6: 0.9910790, 0.9922820, 0.9911456, 0.9923267, -0.0012477, 0.0006773
7: -0.0138900, -0.0062702, -0.0137843, -0.0059755, -0.0058753, 0.0044777
8: -0.0035089, -0.0009761, -0.0033302, -0.0008838, -0.0026251, 0.0014028
9: -0.0053811, -0.0006030, -0.0055653, -0.0006825, -0.0027999, 0.0049623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010006
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010716
time: 1.81 seconds

## BFS IS instance: IS_A1_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058451, 0.0064661, 0.0058572, 0.0065114, -0.0004009, 0.0006088
1: -0.0006208, 0.0006063, -0.0005729, 0.0006940, -0.0013148, 0.0007205
2: 0.0120376, 0.0217693, 0.0113301, 0.0215492, -0.0058115, 0.0104392
3: -0.0044542, -0.0035877, -0.0044372, -0.0035245, -0.0005594, 0.0008495
4: -0.0002746, 0.0039293, -0.0001925, 0.0042358, -0.0027143, 0.0041218
5: -0.0011063, -0.0004588, -0.0011521, -0.0004910, -0.0003759, 0.0006932
6: 0.9910790, 0.9922820, 0.9911537, 0.9923660, -0.0012870, 0.0006895
7: -0.0138900, -0.0062702, -0.0137313, -0.0057152, -0.0062799, 0.0045586
8: -0.0035089, -0.0009761, -0.0033136, -0.0008022, -0.0027067, 0.0014282
9: -0.0053811, -0.0006030, -0.0057281, -0.0007157, -0.0028505, 0.0051250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010031
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010744
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058428, 0.0064739, 0.0058505, 0.0064996, -0.0003837, 0.0006234
1: -0.0006415, 0.0006215, -0.0005860, 0.0006712, -0.0013127, 0.0007207
2: 0.0119152, 0.0218264, 0.0115142, 0.0216549, -0.0058129, 0.0103121
3: -0.0044574, -0.0035768, -0.0044467, -0.0035410, -0.0005354, 0.0008699
4: -0.0002905, 0.0039823, -0.0002383, 0.0041561, -0.0025977, 0.0042206
5: -0.0011142, -0.0004432, -0.0011402, -0.0004842, -0.0003760, 0.0006970
6: 0.9910399, 0.9922967, 0.9911411, 0.9923442, -0.0013043, 0.0006897
7: -0.0139255, -0.0061742, -0.0138142, -0.0058597, -0.0062107, 0.0045597
8: -0.0036171, -0.0009460, -0.0033395, -0.0008474, -0.0027696, 0.0014285
9: -0.0054411, -0.0005719, -0.0056377, -0.0006639, -0.0028512, 0.0050659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010178
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010942
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058428, 0.0064739, 0.0058551, 0.0065212, -0.0004196, 0.0006188
1: -0.0006415, 0.0006215, -0.0005771, 0.0007130, -0.0013545, 0.0007260
2: 0.0119152, 0.0218264, 0.0111771, 0.0215829, -0.0058556, 0.0106492
3: -0.0044574, -0.0035768, -0.0044402, -0.0035109, -0.0005856, 0.0008635
4: -0.0002905, 0.0039823, -0.0002071, 0.0043021, -0.0028410, 0.0041894
5: -0.0011142, -0.0004432, -0.0011620, -0.0004888, -0.0003788, 0.0007188
6: 0.9910399, 0.9922967, 0.9911497, 0.9923842, -0.0013443, 0.0006947
7: -0.0139255, -0.0061742, -0.0137577, -0.0055952, -0.0066174, 0.0045933
8: -0.0036171, -0.0009460, -0.0033218, -0.0007646, -0.0028525, 0.0014390
9: -0.0054411, -0.0005719, -0.0058031, -0.0006992, -0.0028721, 0.0052312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010233
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010963
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058501, 0.0064735, 0.0058585, 0.0065127, -0.0004123, 0.0003732
1: -0.0005868, 0.0006207, -0.0005705, 0.0006967, -0.0007985, 0.0007229
2: 0.0119216, 0.0216609, 0.0113085, 0.0215294, -0.0058304, 0.0064406
3: -0.0044472, -0.0035773, -0.0044355, -0.0035226, -0.0005752, 0.0005207
4: -0.0002409, 0.0039795, -0.0001839, 0.0042452, -0.0027910, 0.0025266
5: -0.0011138, -0.0004838, -0.0011535, -0.0004923, -0.0003772, 0.0004166
6: 0.9911403, 0.9922959, 0.9911560, 0.9923686, -0.0007641, 0.0006917
7: -0.0138189, -0.0061792, -0.0137157, -0.0056983, -0.0050521, 0.0045735
8: -0.0033410, -0.0009476, -0.0033087, -0.0007969, -0.0015828, 0.0014328
9: -0.0054380, -0.0006609, -0.0057387, -0.0007254, -0.0028598, 0.0031591

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010162
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010912
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058501, 0.0064735, 0.0058628, 0.0065334, -0.0004473, 0.0003766
1: -0.0005868, 0.0006207, -0.0005622, 0.0007366, -0.0008664, 0.0007294
2: 0.0119216, 0.0216609, 0.0109864, 0.0214623, -0.0058835, 0.0069885
3: -0.0044472, -0.0035773, -0.0044295, -0.0034938, -0.0006242, 0.0005255
4: -0.0002409, 0.0039795, -0.0001548, 0.0043848, -0.0030284, 0.0025496
5: -0.0011138, -0.0004838, -0.0011743, -0.0004966, -0.0003806, 0.0004521
6: 0.9911403, 0.9922959, 0.9911639, 0.9924068, -0.0008291, 0.0006980
7: -0.0138189, -0.0061792, -0.0136631, -0.0054456, -0.0054819, 0.0046151
8: -0.0033410, -0.0009476, -0.0032922, -0.0007177, -0.0017175, 0.0014459
9: -0.0054380, -0.0006609, -0.0058966, -0.0007583, -0.0028858, 0.0034278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010213
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010934
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058541, 0.0064748, 0.0058690, 0.0065018, -0.0003578, 0.0003471
1: -0.0005790, 0.0006232, -0.0005502, 0.0006756, -0.0006930, 0.0006722
2: 0.0119012, 0.0215979, 0.0114788, 0.0213661, -0.0054223, 0.0055900
3: -0.0044416, -0.0035755, -0.0044209, -0.0035378, -0.0004993, 0.0004843
4: -0.0002136, 0.0039884, -0.0001131, 0.0041714, -0.0024224, 0.0023497
5: -0.0011151, -0.0004879, -0.0011424, -0.0005029, -0.0003508, 0.0003616
6: 0.9911478, 0.9922982, 0.9911754, 0.9923484, -0.0006632, 0.0006433
7: -0.0137695, -0.0061632, -0.0135876, -0.0058319, -0.0043849, 0.0042533
8: -0.0033255, -0.0009426, -0.0032686, -0.0008387, -0.0013738, 0.0013325
9: -0.0054479, -0.0006918, -0.0056551, -0.0008055, -0.0026596, 0.0027418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009959
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058587, 0.0064936, 0.0058690, 0.0065018, -0.0003608, 0.0003706
1: -0.0005701, 0.0006595, -0.0005502, 0.0006756, -0.0006988, 0.0007178
2: 0.0116081, 0.0215263, 0.0114788, 0.0213661, -0.0057898, 0.0056365
3: -0.0044352, -0.0035493, -0.0044209, -0.0035378, -0.0005034, 0.0005171
4: -0.0001826, 0.0041154, -0.0001131, 0.0041714, -0.0024425, 0.0025089
5: -0.0011341, -0.0004925, -0.0011424, -0.0005029, -0.0003745, 0.0003646
6: 0.9911563, 0.9923331, 0.9911754, 0.9923484, -0.0006687, 0.0006869
7: -0.0137133, -0.0059333, -0.0135876, -0.0058319, -0.0044213, 0.0045416
8: -0.0033080, -0.0008705, -0.0032686, -0.0008387, -0.0013852, 0.0014229
9: -0.0055917, -0.0007269, -0.0056551, -0.0008055, -0.0028398, 0.0027646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058515, 0.0064857, 0.0058868, 0.0065015, -0.0003757, 0.0003494
1: -0.0005840, 0.0006444, -0.0005158, 0.0006748, -0.0007278, 0.0006767
2: 0.0117306, 0.0216384, 0.0114847, 0.0210880, -0.0054583, 0.0058702
3: -0.0044452, -0.0035603, -0.0043960, -0.0035383, -0.0005243, 0.0004875
4: -0.0002311, 0.0040623, 0.0000074, 0.0041688, -0.0025438, 0.0023653
5: -0.0011262, -0.0004852, -0.0011421, -0.0005208, -0.0003531, 0.0003797
6: 0.9911430, 0.9923186, 0.9912084, 0.9923477, -0.0006965, 0.0006476
7: -0.0138012, -0.0060294, -0.0133695, -0.0058365, -0.0046047, 0.0042816
8: -0.0033355, -0.0009006, -0.0032002, -0.0008402, -0.0014426, 0.0013414
9: -0.0055316, -0.0006720, -0.0056522, -0.0009419, -0.0026773, 0.0028793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009958
time: 1.22 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009957
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058515, 0.0064857, 0.0058893, 0.0065252, -0.0003758, 0.0003245
1: -0.0005840, 0.0006444, -0.0005109, 0.0007207, -0.0007278, 0.0006286
2: 0.0117306, 0.0216384, 0.0111146, 0.0210484, -0.0050701, 0.0058706
3: -0.0044452, -0.0035603, -0.0043925, -0.0035053, -0.0005243, 0.0004528
4: -0.0002311, 0.0040623, 0.0000245, 0.0043292, -0.0025440, 0.0021971
5: -0.0011262, -0.0004852, -0.0011660, -0.0005234, -0.0003280, 0.0003798
6: 0.9911430, 0.9923186, 0.9912130, 0.9923916, -0.0006965, 0.0006015
7: -0.0138012, -0.0060294, -0.0133384, -0.0055462, -0.0046050, 0.0039771
8: -0.0033355, -0.0009006, -0.0031905, -0.0007492, -0.0014427, 0.0012460
9: -0.0055316, -0.0006720, -0.0058338, -0.0009614, -0.0024868, 0.0028795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009958
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009957
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058535, 0.0064945, 0.0058609, 0.0064816, -0.0003398, 0.0003713
1: -0.0005802, 0.0006613, -0.0005658, 0.0006363, -0.0006581, 0.0007192
2: 0.0115940, 0.0216075, 0.0117954, 0.0214918, -0.0058012, 0.0053083
3: -0.0044424, -0.0035481, -0.0044321, -0.0035661, -0.0004741, 0.0005181
4: -0.0002177, 0.0041215, -0.0001676, 0.0040342, -0.0023003, 0.0025139
5: -0.0011350, -0.0004872, -0.0011220, -0.0004947, -0.0003753, 0.0003434
6: 0.9911467, 0.9923348, 0.9911604, 0.9923108, -0.0006298, 0.0006883
7: -0.0137770, -0.0059222, -0.0136863, -0.0060802, -0.0041639, 0.0045506
8: -0.0033279, -0.0008670, -0.0032995, -0.0009165, -0.0013045, 0.0014257
9: -0.0055986, -0.0006871, -0.0054999, -0.0007438, -0.0028454, 0.0026037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010058
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010058
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058535, 0.0064945, 0.0058679, 0.0065069, -0.0003562, 0.0003579
1: -0.0005802, 0.0006613, -0.0005523, 0.0006854, -0.0006900, 0.0006932
2: 0.0115940, 0.0216075, 0.0113991, 0.0213825, -0.0055909, 0.0055651
3: -0.0044424, -0.0035481, -0.0044223, -0.0035307, -0.0004971, 0.0004993
4: -0.0002177, 0.0041215, -0.0001203, 0.0042059, -0.0024116, 0.0024227
5: -0.0011350, -0.0004872, -0.0011476, -0.0005018, -0.0003617, 0.0003600
6: 0.9911467, 0.9923348, 0.9911734, 0.9923579, -0.0006603, 0.0006633
7: -0.0137770, -0.0059222, -0.0136005, -0.0057694, -0.0043654, 0.0043856
8: -0.0033279, -0.0008670, -0.0032726, -0.0008192, -0.0013676, 0.0013740
9: -0.0055986, -0.0006871, -0.0056942, -0.0007974, -0.0027423, 0.0027296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010112
time: 1.02 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010112
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058622, 0.0065057, 0.0058682, 0.0064815, -0.0003422, 0.0003960
1: -0.0005634, 0.0006831, -0.0005518, 0.0006361, -0.0006629, 0.0007670
2: 0.0114182, 0.0214726, 0.0117974, 0.0213788, -0.0061868, 0.0053466
3: -0.0044304, -0.0035324, -0.0044220, -0.0035662, -0.0004775, 0.0005526
4: -0.0001593, 0.0041977, -0.0001186, 0.0040334, -0.0023169, 0.0026810
5: -0.0011464, -0.0004960, -0.0011218, -0.0005020, -0.0004002, 0.0003459
6: 0.9911627, 0.9923555, 0.9911739, 0.9923106, -0.0006343, 0.0007340
7: -0.0136711, -0.0057843, -0.0135976, -0.0060817, -0.0041940, 0.0048530
8: -0.0032947, -0.0008238, -0.0032717, -0.0009170, -0.0013139, 0.0015204
9: -0.0056849, -0.0007533, -0.0054989, -0.0007993, -0.0030345, 0.0026225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010048
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010048
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058622, 0.0065057, 0.0058751, 0.0065068, -0.0003592, 0.0003780
1: -0.0005634, 0.0006831, -0.0005384, 0.0006852, -0.0006957, 0.0007322
2: 0.0114182, 0.0214726, 0.0114013, 0.0212706, -0.0059059, 0.0056114
3: -0.0044304, -0.0035324, -0.0044124, -0.0035309, -0.0005012, 0.0005275
4: -0.0001593, 0.0041977, -0.0000718, 0.0042050, -0.0024316, 0.0025593
5: -0.0011464, -0.0004960, -0.0011475, -0.0005090, -0.0003820, 0.0003630
6: 0.9911627, 0.9923555, 0.9911866, 0.9923576, -0.0006658, 0.0007007
7: -0.0136711, -0.0057843, -0.0135127, -0.0057710, -0.0044017, 0.0046327
8: -0.0032947, -0.0008238, -0.0032451, -0.0008197, -0.0013790, 0.0014514
9: -0.0056849, -0.0007533, -0.0056932, -0.0008524, -0.0028968, 0.0027523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010103
time: 1.32 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010103
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058699, 0.0065058, 0.0058407, 0.0064726, -0.0006027, 0.0003991
1: -0.0005485, 0.0006832, -0.0006595, 0.0006189, -0.0006534, 0.0013427
2: 0.0114173, 0.0213522, 0.0119360, 0.0218757, -0.0104584, 0.0052705
3: -0.0044196, -0.0035323, -0.0044603, -0.0035786, -0.0008410, 0.0005568
4: -0.0001071, 0.0041981, -0.0003043, 0.0039733, -0.0040804, 0.0027017
5: -0.0011464, -0.0005038, -0.0011129, -0.0004296, -0.0007168, 0.0003409
6: 0.9911770, 0.9923556, 0.9910060, 0.9922941, -0.0006253, 0.0013496
7: -0.0135767, -0.0057836, -0.0139562, -0.0061905, -0.0041343, 0.0062031
8: -0.0032652, -0.0008236, -0.0037106, -0.0009511, -0.0012952, 0.0028869
9: -0.0056853, -0.0008123, -0.0054309, -0.0005449, -0.0051404, 0.0025851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010137
time: 1.10 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010137
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058360, 0.0064756, 0.0058407, 0.0064726, -0.0006366, 0.0006348
1: -0.0007017, 0.0006247, -0.0006595, 0.0006189, -0.0013206, 0.0012842
2: 0.0118890, 0.0219915, 0.0119360, 0.0218757, -0.0099866, 0.0100555
3: -0.0044669, -0.0035744, -0.0044603, -0.0035786, -0.0008883, 0.0008858
4: -0.0003366, 0.0039936, -0.0003043, 0.0039733, -0.0043099, 0.0042979
5: -0.0011159, -0.0003978, -0.0011129, -0.0004296, -0.0006863, 0.0007150
6: 0.9909267, 0.9922997, 0.9910060, 0.9922941, -0.0013674, 0.0012937
7: -0.0140283, -0.0061537, -0.0139562, -0.0061905, -0.0056161, 0.0059889
8: -0.0039302, -0.0009396, -0.0037106, -0.0009511, -0.0029791, 0.0027710
9: -0.0054539, -0.0004817, -0.0054309, -0.0005449, -0.0049090, 0.0049492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010645
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010645
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058773, 0.0065191, 0.0058486, 0.0064722, -0.0005948, 0.0004246
1: -0.0005340, 0.0007089, -0.0005900, 0.0006181, -0.0006564, 0.0012989
2: 0.0112096, 0.0212352, 0.0119421, 0.0216846, -0.0104750, 0.0052942
3: -0.0044092, -0.0035138, -0.0044493, -0.0035792, -0.0008300, 0.0005924
4: -0.0000564, 0.0042881, -0.0002510, 0.0039707, -0.0040271, 0.0028744
5: -0.0011599, -0.0005113, -0.0011125, -0.0004821, -0.0006778, 0.0003425
6: 0.9911909, 0.9923803, 0.9911369, 0.9922934, -0.0006281, 0.0012434
7: -0.0134850, -0.0056207, -0.0138373, -0.0061953, -0.0041529, 0.0065113
8: -0.0032364, -0.0007726, -0.0033484, -0.0009526, -0.0013011, 0.0025758
9: -0.0057872, -0.0008697, -0.0054279, -0.0006492, -0.0051379, 0.0025967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010137
time: 1.17 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010137
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058456, 0.0064860, 0.0058486, 0.0064722, -0.0006266, 0.0006374
1: -0.0006163, 0.0006449, -0.0005900, 0.0006181, -0.0012345, 0.0012349
2: 0.0117263, 0.0217570, 0.0119421, 0.0216846, -0.0099583, 0.0098150
3: -0.0044535, -0.0035599, -0.0044493, -0.0035792, -0.0008743, 0.0008894
4: -0.0002712, 0.0040642, -0.0002510, 0.0039707, -0.0042419, 0.0043152
5: -0.0011264, -0.0004622, -0.0011125, -0.0004821, -0.0006444, 0.0006503
6: 0.9910873, 0.9923190, 0.9911369, 0.9922934, -0.0012061, 0.0011821
7: -0.0138824, -0.0060260, -0.0138373, -0.0061953, -0.0056335, 0.0063121
8: -0.0034857, -0.0008996, -0.0033484, -0.0009526, -0.0025331, 0.0024488
9: -0.0055337, -0.0006097, -0.0054279, -0.0006492, -0.0048845, 0.0048182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010644
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010644
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058767, 0.0065300, 0.0058380, 0.0064822, -0.0006055, 0.0004212
1: -0.0005353, 0.0007301, -0.0006833, 0.0006375, -0.0006638, 0.0014135
2: 0.0110387, 0.0212458, 0.0117858, 0.0219411, -0.0109024, 0.0053540
3: -0.0044101, -0.0034985, -0.0044640, -0.0035652, -0.0008449, 0.0005877
4: -0.0000610, 0.0043621, -0.0003225, 0.0040384, -0.0040994, 0.0028513
5: -0.0011709, -0.0005106, -0.0011226, -0.0004117, -0.0007592, 0.0003463
6: 0.9911897, 0.9924006, 0.9909612, 0.9923120, -0.0006352, 0.0014394
7: -0.0134933, -0.0054867, -0.0139969, -0.0060727, -0.0041998, 0.0064877
8: -0.0032390, -0.0007306, -0.0038346, -0.0009142, -0.0013158, 0.0031040
9: -0.0058710, -0.0008645, -0.0055045, -0.0005092, -0.0053618, 0.0026261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B2_A2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010138
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010231
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064947, 0.0058380, 0.0064822, -0.0006333, 0.0003774
1: -0.0005891, 0.0006618, -0.0006833, 0.0006375, -0.0006896, 0.0013451
2: 0.0115899, 0.0216796, 0.0117858, 0.0219411, -0.0103513, 0.0055621
3: -0.0044489, -0.0035477, -0.0044640, -0.0035652, -0.0008837, 0.0005266
4: -0.0002490, 0.0041233, -0.0003225, 0.0040384, -0.0042874, 0.0025551
5: -0.0011353, -0.0004826, -0.0011226, -0.0004117, -0.0007236, 0.0003598
6: 0.9911382, 0.9923353, 0.9909612, 0.9923120, -0.0006599, 0.0013741
7: -0.0138335, -0.0059190, -0.0139969, -0.0060727, -0.0043630, 0.0060403
8: -0.0033456, -0.0008660, -0.0038346, -0.0009142, -0.0013669, 0.0029686
9: -0.0056007, -0.0006518, -0.0055045, -0.0005092, -0.0050914, 0.0027281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010760
time: 1.15 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010898
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058847, 0.0065429, 0.0058459, 0.0064818, -0.0005970, 0.0004460
1: -0.0005197, 0.0007551, -0.0006136, 0.0006367, -0.0006674, 0.0013687
2: 0.0108370, 0.0211198, 0.0117922, 0.0217496, -0.0109125, 0.0053830
3: -0.0043989, -0.0034805, -0.0044530, -0.0035658, -0.0008331, 0.0006224
4: -0.0000064, 0.0044495, -0.0002691, 0.0040356, -0.0040420, 0.0030196
5: -0.0011840, -0.0005188, -0.0011222, -0.0004642, -0.0007197, 0.0003482
6: 0.9912045, 0.9924245, 0.9910924, 0.9923112, -0.0006387, 0.0013321
7: -0.0133944, -0.0053285, -0.0138777, -0.0060777, -0.0042225, 0.0067531
8: -0.0032080, -0.0006810, -0.0034715, -0.0009158, -0.0013229, 0.0027905
9: -0.0059699, -0.0009263, -0.0055014, -0.0006138, -0.0053561, 0.0026403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010136
time: 1.29 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010231
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058587, 0.0065060, 0.0058459, 0.0064818, -0.0006231, 0.0004040
1: -0.0005702, 0.0006836, -0.0006136, 0.0006367, -0.0006930, 0.0012972
2: 0.0114142, 0.0215273, 0.0117922, 0.0217496, -0.0103354, 0.0055899
3: -0.0044353, -0.0035320, -0.0044530, -0.0035658, -0.0008695, 0.0005638
4: -0.0001830, 0.0041994, -0.0002691, 0.0040356, -0.0042186, 0.0027354
5: -0.0011466, -0.0004924, -0.0011222, -0.0004642, -0.0006824, 0.0003616
6: 0.9911562, 0.9923561, 0.9910924, 0.9923112, -0.0006632, 0.0012637
7: -0.0137141, -0.0057812, -0.0138777, -0.0060777, -0.0043848, 0.0063449
8: -0.0033082, -0.0008229, -0.0034715, -0.0009158, -0.0013737, 0.0026486
9: -0.0056868, -0.0007264, -0.0055014, -0.0006138, -0.0050730, 0.0027418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010758
time: 1.18 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010895
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0058608, 0.0064921, 0.0058749, 0.0065007, -0.0003703, 0.0003520
1: -0.0005661, 0.0006567, -0.0005387, 0.0006734, -0.0007172, 0.0006817
2: 0.0116308, 0.0214936, 0.0114963, 0.0212731, -0.0054987, 0.0057848
3: -0.0044323, -0.0035514, -0.0044126, -0.0035394, -0.0005167, 0.0004911
4: -0.0001684, 0.0041056, -0.0000729, 0.0041638, -0.0025068, 0.0023828
5: -0.0011326, -0.0004946, -0.0011413, -0.0005089, -0.0003557, 0.0003742
6: 0.9911602, 0.9923304, 0.9911864, 0.9923464, -0.0006863, 0.0006524
7: -0.0136877, -0.0059511, -0.0135147, -0.0058456, -0.0045377, 0.0043133
8: -0.0032999, -0.0008761, -0.0032457, -0.0008430, -0.0014216, 0.0013513
9: -0.0055806, -0.0007430, -0.0056466, -0.0008511, -0.0026971, 0.0028374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008952, upper bound: 0.0008620
time: 1.06 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008476, upper bound: 0.0008141
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0058608, 0.0064921, 0.0058791, 0.0065245, -0.0003674, 0.0003292
1: -0.0005661, 0.0006567, -0.0005306, 0.0007195, -0.0007115, 0.0006376
2: 0.0116308, 0.0214936, 0.0111245, 0.0212080, -0.0051432, 0.0057392
3: -0.0044323, -0.0035514, -0.0044068, -0.0035062, -0.0005126, 0.0004594
4: -0.0001684, 0.0041056, -0.0000446, 0.0043249, -0.0024870, 0.0022287
5: -0.0011326, -0.0004946, -0.0011654, -0.0005131, -0.0003327, 0.0003713
6: 0.9911602, 0.9923304, 0.9911941, 0.9923905, -0.0006809, 0.0006102
7: -0.0136877, -0.0059511, -0.0134636, -0.0055539, -0.0045019, 0.0040344
8: -0.0032999, -0.0008761, -0.0032297, -0.0007517, -0.0014104, 0.0012640
9: -0.0055806, -0.0007430, -0.0058289, -0.0008831, -0.0025227, 0.0028150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_B1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008952, upper bound: 0.0008620
time: 0.91 seconds

## Relational analysis of IS_A2_B2_B1_B1_A1_B2_B2

### Relational analysis result of IS_A2_B2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008476, upper bound: 0.0008141
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058686, 0.0065029, 0.0058821, 0.0065006, -0.0003716, 0.0003741
1: -0.0005510, 0.0006775, -0.0005247, 0.0006732, -0.0007198, 0.0007246
2: 0.0114630, 0.0213720, 0.0114983, 0.0211603, -0.0058447, 0.0058061
3: -0.0044214, -0.0035364, -0.0044025, -0.0035395, -0.0005186, 0.0005220
4: -0.0001157, 0.0041783, -0.0000240, 0.0041630, -0.0025160, 0.0025327
5: -0.0011435, -0.0005025, -0.0011412, -0.0005162, -0.0003781, 0.0003756
6: 0.9911746, 0.9923502, 0.9911997, 0.9923460, -0.0006889, 0.0006934
7: -0.0135922, -0.0058195, -0.0134262, -0.0058471, -0.0045544, 0.0045847
8: -0.0032700, -0.0008349, -0.0032180, -0.0008435, -0.0014269, 0.0014363
9: -0.0056629, -0.0008026, -0.0056456, -0.0009065, -0.0028668, 0.0028478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008838, upper bound: 0.0008593
time: 1.05 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008349, upper bound: 0.0008111
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058686, 0.0065029, 0.0058858, 0.0065244, -0.0003699, 0.0003488
1: -0.0005510, 0.0006775, -0.0005177, 0.0007193, -0.0007164, 0.0006756
2: 0.0114630, 0.0213720, 0.0111265, 0.0211038, -0.0054492, 0.0057788
3: -0.0044214, -0.0035364, -0.0043975, -0.0035063, -0.0005161, 0.0004867
4: -0.0001157, 0.0041783, 0.0000005, 0.0043241, -0.0025042, 0.0023613
5: -0.0011435, -0.0005025, -0.0011652, -0.0005198, -0.0003525, 0.0003738
6: 0.9911746, 0.9923502, 0.9912064, 0.9923902, -0.0006856, 0.0006465
7: -0.0135922, -0.0058195, -0.0133819, -0.0055555, -0.0045330, 0.0042744
8: -0.0032700, -0.0008349, -0.0032041, -0.0007522, -0.0014201, 0.0013391
9: -0.0056629, -0.0008026, -0.0058279, -0.0009342, -0.0026728, 0.0028344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009691, upper bound: 0.0009249
time: 1.43 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009403, upper bound: 0.0009265
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058478, 0.0064751, 0.0058730, 0.0065250, -0.0004086, 0.0003522
1: -0.0005913, 0.0006237, -0.0005424, 0.0007204, -0.0007914, 0.0006821
2: 0.0118972, 0.0216972, 0.0111170, 0.0213032, -0.0055018, 0.0063834
3: -0.0044505, -0.0035752, -0.0044153, -0.0035055, -0.0005701, 0.0004914
4: -0.0002566, 0.0039901, -0.0000859, 0.0043282, -0.0027662, 0.0023841
5: -0.0011154, -0.0004814, -0.0011658, -0.0005069, -0.0003559, 0.0004129
6: 0.9911361, 0.9922988, 0.9911829, 0.9923914, -0.0007573, 0.0006527
7: -0.0138474, -0.0061601, -0.0135383, -0.0055481, -0.0050072, 0.0043157
8: -0.0033499, -0.0009416, -0.0032531, -0.0007498, -0.0015687, 0.0013521
9: -0.0054499, -0.0006431, -0.0058326, -0.0008364, -0.0026986, 0.0031310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0009986
time: 1.10 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0009986
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058598, 0.0064942, 0.0058730, 0.0065250, -0.0003818, 0.0003611
1: -0.0005680, 0.0006608, -0.0005424, 0.0007204, -0.0007396, 0.0006995
2: 0.0115983, 0.0215089, 0.0111170, 0.0213032, -0.0056419, 0.0059653
3: -0.0044336, -0.0035485, -0.0044153, -0.0035055, -0.0005328, 0.0005039
4: -0.0001750, 0.0041196, -0.0000859, 0.0043282, -0.0025850, 0.0024448
5: -0.0011347, -0.0004936, -0.0011658, -0.0005069, -0.0003650, 0.0003859
6: 0.9911584, 0.9923342, 0.9911829, 0.9923914, -0.0007077, 0.0006694
7: -0.0136997, -0.0059256, -0.0135383, -0.0055481, -0.0046793, 0.0044256
8: -0.0033037, -0.0008681, -0.0032531, -0.0007498, -0.0014660, 0.0013865
9: -0.0055965, -0.0007355, -0.0058326, -0.0008364, -0.0027673, 0.0029259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010151
time: 1.05 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010151
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058570, 0.0064855, 0.0058797, 0.0065249, -0.0004112, 0.0003724
1: -0.0005734, 0.0006438, -0.0005294, 0.0007202, -0.0007965, 0.0007214
2: 0.0117349, 0.0215531, 0.0111191, 0.0211978, -0.0058186, 0.0064246
3: -0.0044376, -0.0035607, -0.0044058, -0.0035057, -0.0005738, 0.0005197
4: -0.0001942, 0.0040605, -0.0000402, 0.0043273, -0.0027840, 0.0025214
5: -0.0011259, -0.0004908, -0.0011657, -0.0005137, -0.0003764, 0.0004156
6: 0.9911531, 0.9923180, 0.9911953, 0.9923910, -0.0007622, 0.0006903
7: -0.0137344, -0.0060327, -0.0134556, -0.0055497, -0.0050396, 0.0045642
8: -0.0033145, -0.0009017, -0.0032272, -0.0007503, -0.0015789, 0.0014299
9: -0.0055295, -0.0007138, -0.0058316, -0.0008881, -0.0028540, 0.0031512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0009984
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0009984
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058677, 0.0065054, 0.0058797, 0.0065249, -0.0003846, 0.0003814
1: -0.0005528, 0.0006826, -0.0005294, 0.0007202, -0.0007450, 0.0007387
2: 0.0114225, 0.0213865, 0.0111191, 0.0211978, -0.0059581, 0.0060092
3: -0.0044227, -0.0035328, -0.0044058, -0.0035057, -0.0005367, 0.0005321
4: -0.0001220, 0.0041958, -0.0000402, 0.0043273, -0.0026040, 0.0025819
5: -0.0011461, -0.0005015, -0.0011657, -0.0005137, -0.0003854, 0.0003887
6: 0.9911729, 0.9923550, 0.9911953, 0.9923910, -0.0007129, 0.0007069
7: -0.0136037, -0.0057877, -0.0134556, -0.0055497, -0.0047137, 0.0046736
8: -0.0032736, -0.0008249, -0.0032272, -0.0007503, -0.0014768, 0.0014642
9: -0.0056827, -0.0007955, -0.0058316, -0.0008881, -0.0029224, 0.0029474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010137
time: 1.35 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010137
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058762, 0.0065255, 0.0058388, 0.0064737, -0.0005975, 0.0004306
1: -0.0005362, 0.0007214, -0.0006764, 0.0006211, -0.0006598, 0.0013978
2: 0.0111095, 0.0212525, 0.0119182, 0.0219221, -0.0108126, 0.0053216
3: -0.0044107, -0.0035048, -0.0044629, -0.0035770, -0.0008337, 0.0006008
4: -0.0000639, 0.0043315, -0.0003172, 0.0039810, -0.0040449, 0.0029149
5: -0.0011663, -0.0005102, -0.0011140, -0.0004169, -0.0007495, 0.0003442
6: 0.9911888, 0.9923922, 0.9909743, 0.9922963, -0.0006314, 0.0014179
7: -0.0134985, -0.0055421, -0.0139851, -0.0061765, -0.0041744, 0.0065423
8: -0.0032407, -0.0007480, -0.0037986, -0.0009467, -0.0013078, 0.0030506
9: -0.0058363, -0.0008612, -0.0054396, -0.0005196, -0.0053167, 0.0026102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010217
time: 1.29 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010060
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058488, 0.0064925, 0.0058388, 0.0064737, -0.0006249, 0.0003865
1: -0.0005893, 0.0006575, -0.0006764, 0.0006211, -0.0006784, 0.0013339
2: 0.0116243, 0.0216814, 0.0119182, 0.0219221, -0.0102978, 0.0054723
3: -0.0044490, -0.0035508, -0.0044629, -0.0035770, -0.0008720, 0.0005394
4: -0.0002498, 0.0041084, -0.0003172, 0.0039810, -0.0042308, 0.0026169
5: -0.0011330, -0.0004825, -0.0011140, -0.0004169, -0.0007162, 0.0003540
6: 0.9911379, 0.9923311, 0.9909743, 0.9922963, -0.0006492, 0.0013568
7: -0.0138350, -0.0059460, -0.0139851, -0.0061765, -0.0042925, 0.0061366
8: -0.0033461, -0.0008745, -0.0037986, -0.0009467, -0.0013448, 0.0029241
9: -0.0055838, -0.0006509, -0.0054396, -0.0005196, -0.0050642, 0.0026841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010784
time: 1.30 seconds

## Relational analysis of IS_A2_B2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010784
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058833, 0.0065253, 0.0058471, 0.0064868, -0.0006035, 0.0004331
1: -0.0005226, 0.0007211, -0.0006036, 0.0006464, -0.0006956, 0.0013247
2: 0.0111117, 0.0211429, 0.0117137, 0.0217221, -0.0106105, 0.0056107
3: -0.0044009, -0.0035050, -0.0044515, -0.0035588, -0.0008422, 0.0006043
4: -0.0000164, 0.0043305, -0.0002615, 0.0040696, -0.0040860, 0.0029322
5: -0.0011662, -0.0005173, -0.0011272, -0.0004718, -0.0006944, 0.0003629
6: 0.9912018, 0.9923919, 0.9911112, 0.9923205, -0.0006657, 0.0012807
7: -0.0134126, -0.0055439, -0.0138607, -0.0060161, -0.0044011, 0.0065915
8: -0.0032137, -0.0007485, -0.0034195, -0.0008965, -0.0013788, 0.0026710
9: -0.0058352, -0.0009150, -0.0055399, -0.0006288, -0.0052065, 0.0027520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010031
time: 1.81 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010171
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058566, 0.0064924, 0.0058471, 0.0064868, -0.0006302, 0.0003893
1: -0.0005742, 0.0006573, -0.0006036, 0.0006464, -0.0007305, 0.0012609
2: 0.0116263, 0.0215594, 0.0117137, 0.0217221, -0.0100958, 0.0058919
3: -0.0044381, -0.0035510, -0.0044515, -0.0035588, -0.0008794, 0.0005432
4: -0.0001969, 0.0041075, -0.0002615, 0.0040696, -0.0042665, 0.0026356
5: -0.0011329, -0.0004903, -0.0011272, -0.0004718, -0.0006611, 0.0003811
6: 0.9911523, 0.9923308, 0.9911112, 0.9923205, -0.0006990, 0.0012196
7: -0.0137393, -0.0059476, -0.0138607, -0.0060161, -0.0046217, 0.0061662
8: -0.0033161, -0.0008750, -0.0034195, -0.0008965, -0.0014480, 0.0025445
9: -0.0055828, -0.0007107, -0.0055399, -0.0006288, -0.0049540, 0.0028899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010752
time: 1.24 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010752
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0058812, 0.0065363, 0.0058443, 0.0064922, -0.0006110, 0.0004370
1: -0.0005266, 0.0007422, -0.0006283, 0.0006569, -0.0006741, 0.0013706
2: 0.0109411, 0.0211751, 0.0116297, 0.0217900, -0.0108490, 0.0054369
3: -0.0044038, -0.0034898, -0.0044554, -0.0035513, -0.0008525, 0.0006098
4: -0.0000304, 0.0044044, -0.0002804, 0.0041060, -0.0041364, 0.0029587
5: -0.0011772, -0.0005152, -0.0011327, -0.0004531, -0.0007241, 0.0003517
6: 0.9911979, 0.9924121, 0.9910648, 0.9923305, -0.0006451, 0.0013474
7: -0.0134378, -0.0054100, -0.0139029, -0.0059502, -0.0042648, 0.0066992
8: -0.0032216, -0.0007066, -0.0035482, -0.0008758, -0.0013361, 0.0028416
9: -0.0059189, -0.0008992, -0.0055811, -0.0005917, -0.0053272, 0.0026667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010264
time: 1.17 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010264
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0058533, 0.0065021, 0.0058443, 0.0064922, -0.0006389, 0.0004042
1: -0.0005806, 0.0006760, -0.0006283, 0.0006569, -0.0006906, 0.0013043
2: 0.0114755, 0.0216112, 0.0116297, 0.0217900, -0.0103145, 0.0055706
3: -0.0044428, -0.0035375, -0.0044554, -0.0035513, -0.0008915, 0.0005640
4: -0.0002194, 0.0041729, -0.0002804, 0.0041060, -0.0043254, 0.0027364
5: -0.0011427, -0.0004870, -0.0011327, -0.0004531, -0.0006895, 0.0003604
6: 0.9911462, 0.9923488, 0.9910648, 0.9923305, -0.0006609, 0.0012840
7: -0.0137799, -0.0058293, -0.0139029, -0.0059502, -0.0043697, 0.0064736
8: -0.0033288, -0.0008379, -0.0035482, -0.0008758, -0.0013690, 0.0027103
9: -0.0056568, -0.0006853, -0.0055811, -0.0005917, -0.0050651, 0.0027323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010880
time: 1.13 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010881
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0058889, 0.0065493, 0.0058515, 0.0064920, -0.0003446, 0.0004632
1: -0.0005117, 0.0007675, -0.0005841, 0.0006565, -0.0006675, 0.0008972
2: 0.0107372, 0.0210549, 0.0116325, 0.0216395, -0.0072370, 0.0053842
3: -0.0043931, -0.0034716, -0.0044453, -0.0035515, -0.0004809, 0.0006464
4: 0.0000217, 0.0044928, -0.0002316, 0.0041048, -0.0023332, 0.0031361
5: -0.0011904, -0.0005230, -0.0011325, -0.0004852, -0.0004681, 0.0003483
6: 0.9912123, 0.9924364, 0.9911429, 0.9923302, -0.0006388, 0.0008586
7: -0.0133436, -0.0052502, -0.0138021, -0.0059524, -0.0042234, 0.0056768
8: -0.0031921, -0.0006565, -0.0033358, -0.0008765, -0.0013232, 0.0017785
9: -0.0060189, -0.0009581, -0.0055798, -0.0006714, -0.0035496, 0.0026409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010208
time: 1.15 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010289
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0058627, 0.0065132, 0.0058515, 0.0064920, -0.0003519, 0.0004318
1: -0.0005623, 0.0006975, -0.0005841, 0.0006565, -0.0006817, 0.0008364
2: 0.0113020, 0.0214637, 0.0116325, 0.0216395, -0.0067464, 0.0054981
3: -0.0044296, -0.0035220, -0.0044453, -0.0035515, -0.0004911, 0.0006026
4: -0.0001555, 0.0042480, -0.0002316, 0.0041048, -0.0023826, 0.0029235
5: -0.0011539, -0.0004965, -0.0011325, -0.0004852, -0.0004364, 0.0003557
6: 0.9911638, 0.9923694, 0.9911429, 0.9923302, -0.0006523, 0.0008004
7: -0.0136642, -0.0056932, -0.0138021, -0.0059524, -0.0043128, 0.0052920
8: -0.0032926, -0.0007953, -0.0033358, -0.0008765, -0.0013512, 0.0016580
9: -0.0057419, -0.0007576, -0.0055798, -0.0006714, -0.0033090, 0.0026968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010744
time: 1.43 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010933
time: 1.15 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.37 seconds
IS_A1_B1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009982
IS_A1_B1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009982
IS_A1_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009998
IS_A1_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010229, upper bound: 0.0009998
IS_A1_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009980
IS_A1_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009980
IS_A1_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009996
IS_A1_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010169, upper bound: 0.0009996
IS_A1_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010069
IS_A1_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010121
IS_A1_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010081
IS_A1_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010271, upper bound: 0.0010168
IS_A1_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010055
IS_A1_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010111
IS_A1_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010066
IS_A1_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010211, upper bound: 0.0010161
IS_A1_B1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010128
IS_A1_B1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010734
IS_A1_B1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010138
IS_A1_B1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010027, upper bound: 0.0010760
IS_A1_B1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010127
IS_A1_B1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010733
IS_A1_B1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010136
IS_A1_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010758
IS_A1_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010214
IS_A1_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010869
IS_A1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010231
IS_A1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010898
IS_A1_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010213
IS_A1_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010865
IS_A1_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010231
IS_A1_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010894
IS_A1_B2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008508, upper bound: 0.0008727
IS_A1_B2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008236, upper bound: 0.0008540
IS_A1_B2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009243, upper bound: 0.0009729
IS_A1_B2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009256, upper bound: 0.0009449
IS_A1_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008533, upper bound: 0.0008619
IS_A1_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008197, upper bound: 0.0008339
IS_A1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008427, upper bound: 0.0008569
IS_A1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008114, upper bound: 0.0008304
IS_A1_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010258
IS_A1_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010318
IS_A1_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010267
IS_A1_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010344
IS_A1_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010195
IS_A1_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010264
IS_A1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010208
IS_A1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010289
IS_A1_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010011
IS_A1_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010716
IS_A1_B2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010035
IS_A1_B2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010744
IS_A1_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010006
IS_A1_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010716
IS_A1_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010031
IS_A1_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010744
IS_A1_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010178
IS_A1_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010942
IS_A1_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010233
IS_A1_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010963
IS_A1_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010162
IS_A1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010912
IS_A1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010213
IS_A1_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010934
IS_A2_B1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009959
IS_A2_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
IS_A2_B1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
IS_A2_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
IS_A2_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009958
IS_A2_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009957
IS_A2_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009958
IS_A2_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009957
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010058
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010058
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010112
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010112
IS_A2_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010048
IS_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010048
IS_A2_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010103
IS_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010103
IS_A2_B1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010137
IS_A2_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010137
IS_A2_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010645
IS_A2_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010645
IS_A2_B1_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010137
IS_A2_B1_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010137
IS_A2_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010644
IS_A2_B1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010644
IS_A2_B1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010138
IS_A2_B1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010231
IS_A2_B1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010760
IS_A2_B1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010898
IS_A2_B1_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010136
IS_A2_B1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010231
IS_A2_B1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010758
IS_A2_B1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010895
IS_A2_B2_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008952, upper bound: 0.0008620
IS_A2_B2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008476, upper bound: 0.0008141
IS_A2_B2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008952, upper bound: 0.0008620
IS_A2_B2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008476, upper bound: 0.0008141
IS_A2_B2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008838, upper bound: 0.0008593
IS_A2_B2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0008349, upper bound: 0.0008111
IS_A2_B2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009691, upper bound: 0.0009249
IS_A2_B2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0009403, upper bound: 0.0009265
IS_A2_B2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0009986
IS_A2_B2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0009986
IS_A2_B2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010151
IS_A2_B2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010151
IS_A2_B2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0009984
IS_A2_B2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0009984
IS_A2_B2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010137
IS_A2_B2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010137
IS_A2_B2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010217
IS_A2_B2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010060
IS_A2_B2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010784
IS_A2_B2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010784
IS_A2_B2_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010031
IS_A2_B2_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010171
IS_A2_B2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010752
IS_A2_B2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010752
IS_A2_B2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010264
IS_A2_B2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010264
IS_A2_B2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010880
IS_A2_B2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010881
IS_A2_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010208
IS_A2_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010289
IS_A2_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010744
IS_A2_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010933

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058494, 0.0064547, 0.0058794, 0.0065016, -0.0003544, 0.0003017
1: -0.0005881, 0.0005842, -0.0005301, 0.0006751, -0.0006865, 0.0005844
2: 0.0122157, 0.0216712, 0.0114827, 0.0212036, -0.0047135, 0.0055374
3: -0.0044481, -0.0036036, -0.0044064, -0.0035381, -0.0004946, 0.0004210
4: -0.0002454, 0.0038521, -0.0000427, 0.0041697, -0.0023996, 0.0020425
5: -0.0010948, -0.0004831, -0.0011422, -0.0005134, -0.0003049, 0.0003582
6: 0.9911391, 0.9922610, 0.9911945, 0.9923480, -0.0006570, 0.0005592
7: -0.0138270, -0.0064099, -0.0134602, -0.0058349, -0.0043437, 0.0036974
8: -0.0033436, -0.0010198, -0.0032286, -0.0008397, -0.0013608, 0.0011584
9: -0.0052937, -0.0006559, -0.0056532, -0.0008852, -0.0023119, 0.0027160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008825, upper bound: 0.0008542
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008657, upper bound: 0.0008273
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058544, 0.0064732, 0.0058794, 0.0065016, -0.0003571, 0.0003249
1: -0.0005785, 0.0006201, -0.0005301, 0.0006751, -0.0006916, 0.0006293
2: 0.0119266, 0.0215942, 0.0114827, 0.0212036, -0.0050761, 0.0055782
3: -0.0044413, -0.0035778, -0.0044064, -0.0035381, -0.0004982, 0.0004534
4: -0.0002120, 0.0039774, -0.0000427, 0.0041697, -0.0024173, 0.0021997
5: -0.0011135, -0.0004881, -0.0011422, -0.0005134, -0.0003284, 0.0003608
6: 0.9911482, 0.9922953, 0.9911945, 0.9923480, -0.0006618, 0.0006023
7: -0.0137666, -0.0061831, -0.0134602, -0.0058349, -0.0043757, 0.0039818
8: -0.0033246, -0.0009488, -0.0032286, -0.0008397, -0.0013709, 0.0012475
9: -0.0054355, -0.0006936, -0.0056532, -0.0008852, -0.0024898, 0.0027361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008825, upper bound: 0.0008542
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008657, upper bound: 0.0008273
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058494, 0.0064547, 0.0058824, 0.0065253, -0.0003976, 0.0003142
1: -0.0005881, 0.0005842, -0.0005243, 0.0007210, -0.0007701, 0.0006086
2: 0.0122157, 0.0216712, 0.0111124, 0.0211570, -0.0049088, 0.0062118
3: -0.0044481, -0.0036036, -0.0044022, -0.0035051, -0.0005548, 0.0004384
4: -0.0002454, 0.0038521, -0.0000225, 0.0043302, -0.0026918, 0.0021272
5: -0.0010948, -0.0004831, -0.0011661, -0.0005164, -0.0003175, 0.0004018
6: 0.9911391, 0.9922610, 0.9912001, 0.9923919, -0.0007370, 0.0005824
7: -0.0138270, -0.0064099, -0.0134237, -0.0055444, -0.0048726, 0.0038505
8: -0.0033436, -0.0010198, -0.0032172, -0.0007487, -0.0015266, 0.0012063
9: -0.0052937, -0.0006559, -0.0058349, -0.0009081, -0.0024077, 0.0030468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008708, upper bound: 0.0008430
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008548, upper bound: 0.0008182
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058544, 0.0064732, 0.0058824, 0.0065253, -0.0004002, 0.0003374
1: -0.0005785, 0.0006201, -0.0005243, 0.0007210, -0.0007752, 0.0006535
2: 0.0119266, 0.0215942, 0.0111124, 0.0211570, -0.0052714, 0.0062526
3: -0.0044413, -0.0035778, -0.0044022, -0.0035051, -0.0005584, 0.0004708
4: -0.0002120, 0.0039774, -0.0000225, 0.0043302, -0.0027095, 0.0022843
5: -0.0011135, -0.0004881, -0.0011661, -0.0005164, -0.0003410, 0.0004045
6: 0.9911482, 0.9922953, 0.9912001, 0.9923919, -0.0007418, 0.0006254
7: -0.0137666, -0.0061831, -0.0134237, -0.0055444, -0.0049046, 0.0041350
8: -0.0033246, -0.0009488, -0.0032172, -0.0007487, -0.0015366, 0.0012955
9: -0.0054355, -0.0006936, -0.0058349, -0.0009081, -0.0025856, 0.0030668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008708, upper bound: 0.0008430
time: 0.92 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008548, upper bound: 0.0008182
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058590, 0.0064656, 0.0058868, 0.0065015, -0.0003561, 0.0003188
1: -0.0005695, 0.0006054, -0.0005158, 0.0006748, -0.0006897, 0.0006175
2: 0.0120447, 0.0215216, 0.0114847, 0.0210880, -0.0049810, 0.0055633
3: -0.0044348, -0.0035883, -0.0043960, -0.0035383, -0.0004969, 0.0004449
4: -0.0001805, 0.0039262, 0.0000074, 0.0041688, -0.0024108, 0.0021585
5: -0.0011058, -0.0004928, -0.0011421, -0.0005208, -0.0003222, 0.0003599
6: 0.9911569, 0.9922812, 0.9912084, 0.9923477, -0.0006600, 0.0005910
7: -0.0137097, -0.0062758, -0.0133695, -0.0058365, -0.0043639, 0.0039072
8: -0.0033068, -0.0009778, -0.0032002, -0.0008402, -0.0013672, 0.0012241
9: -0.0053775, -0.0007292, -0.0056522, -0.0009419, -0.0024431, 0.0027287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008766, upper bound: 0.0008536
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008579, upper bound: 0.0008264
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058627, 0.0064863, 0.0058868, 0.0065015, -0.0003584, 0.0003436
1: -0.0005623, 0.0006454, -0.0005158, 0.0006748, -0.0006942, 0.0006656
2: 0.0117219, 0.0214636, 0.0114847, 0.0210880, -0.0053686, 0.0055996
3: -0.0044296, -0.0035595, -0.0043960, -0.0035383, -0.0005001, 0.0004795
4: -0.0001554, 0.0040661, 0.0000074, 0.0041688, -0.0024265, 0.0023264
5: -0.0011267, -0.0004965, -0.0011421, -0.0005208, -0.0003473, 0.0003622
6: 0.9911637, 0.9923195, 0.9912084, 0.9923477, -0.0006644, 0.0006370
7: -0.0136641, -0.0060226, -0.0133695, -0.0058365, -0.0043924, 0.0042113
8: -0.0032925, -0.0008985, -0.0032002, -0.0008402, -0.0013761, 0.0013194
9: -0.0055359, -0.0007577, -0.0056522, -0.0009419, -0.0026333, 0.0027465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009726, upper bound: 0.0009204
time: 1.02 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009439, upper bound: 0.0009215
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058590, 0.0064656, 0.0058893, 0.0065252, -0.0003993, 0.0003307
1: -0.0005695, 0.0006054, -0.0005109, 0.0007207, -0.0007733, 0.0006405
2: 0.0120447, 0.0215216, 0.0111146, 0.0210484, -0.0051663, 0.0062375
3: -0.0044348, -0.0035883, -0.0043925, -0.0035053, -0.0005571, 0.0004614
4: -0.0001805, 0.0039262, 0.0000245, 0.0043292, -0.0027029, 0.0022388
5: -0.0011058, -0.0004928, -0.0011660, -0.0005234, -0.0003342, 0.0004035
6: 0.9911569, 0.9922812, 0.9912130, 0.9923916, -0.0007400, 0.0006130
7: -0.0137097, -0.0062758, -0.0133384, -0.0055462, -0.0048928, 0.0040526
8: -0.0033068, -0.0009778, -0.0031905, -0.0007492, -0.0015329, 0.0012696
9: -0.0053775, -0.0007292, -0.0058338, -0.0009614, -0.0025340, 0.0030594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008658, upper bound: 0.0008425
time: 1.01 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008471, upper bound: 0.0008170
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058627, 0.0064863, 0.0058893, 0.0065252, -0.0004016, 0.0003555
1: -0.0005623, 0.0006454, -0.0005109, 0.0007207, -0.0007778, 0.0006886
2: 0.0117219, 0.0214636, 0.0111146, 0.0210484, -0.0055540, 0.0062737
3: -0.0044296, -0.0035595, -0.0043925, -0.0035053, -0.0005603, 0.0004961
4: -0.0001554, 0.0040661, 0.0000245, 0.0043292, -0.0027187, 0.0024068
5: -0.0011267, -0.0004965, -0.0011660, -0.0005234, -0.0003593, 0.0004058
6: 0.9911637, 0.9923195, 0.9912130, 0.9923916, -0.0007443, 0.0006589
7: -0.0136641, -0.0060226, -0.0133384, -0.0055462, -0.0049212, 0.0043567
8: -0.0032925, -0.0008985, -0.0031905, -0.0007492, -0.0015418, 0.0013649
9: -0.0055359, -0.0007577, -0.0058338, -0.0009614, -0.0027242, 0.0030772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009703, upper bound: 0.0009199
time: 0.99 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009414, upper bound: 0.0009204
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064734, 0.0058710, 0.0064813, -0.0003362, 0.0003320
1: -0.0005892, 0.0006204, -0.0005463, 0.0006358, -0.0006512, 0.0006431
2: 0.0119238, 0.0216800, 0.0117993, 0.0213344, -0.0051872, 0.0052529
3: -0.0044489, -0.0035775, -0.0044181, -0.0035664, -0.0004692, 0.0004633
4: -0.0002492, 0.0039786, -0.0000994, 0.0040325, -0.0022763, 0.0022478
5: -0.0011137, -0.0004825, -0.0011217, -0.0005049, -0.0003356, 0.0003398
6: 0.9911380, 0.9922956, 0.9911790, 0.9923103, -0.0006232, 0.0006154
7: -0.0138339, -0.0061809, -0.0135628, -0.0060833, -0.0041205, 0.0040689
8: -0.0033457, -0.0009481, -0.0032608, -0.0009175, -0.0012909, 0.0012748
9: -0.0054369, -0.0006516, -0.0054979, -0.0008211, -0.0025443, 0.0025765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010087
time: 1.18 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010087
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064734, 0.0058784, 0.0065067, -0.0003522, 0.0003222
1: -0.0005892, 0.0006204, -0.0005320, 0.0006850, -0.0006821, 0.0006241
2: 0.0119238, 0.0216800, 0.0114028, 0.0212187, -0.0050341, 0.0055017
3: -0.0044489, -0.0035775, -0.0044077, -0.0035310, -0.0004914, 0.0004496
4: -0.0002492, 0.0039786, -0.0000493, 0.0042043, -0.0023841, 0.0021815
5: -0.0011137, -0.0004825, -0.0011474, -0.0005124, -0.0003256, 0.0003559
6: 0.9911380, 0.9922956, 0.9911928, 0.9923575, -0.0006527, 0.0005973
7: -0.0138339, -0.0061809, -0.0134720, -0.0057723, -0.0043156, 0.0039488
8: -0.0033457, -0.0009481, -0.0032323, -0.0008201, -0.0013521, 0.0012371
9: -0.0054369, -0.0006516, -0.0056924, -0.0008778, -0.0024692, 0.0026985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010129
time: 1.16 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010129
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064734, 0.0058746, 0.0065055, -0.0003790, 0.0003442
1: -0.0005892, 0.0006204, -0.0005392, 0.0006826, -0.0007342, 0.0006666
2: 0.0119238, 0.0216800, 0.0114224, 0.0212774, -0.0053771, 0.0059216
3: -0.0044489, -0.0035775, -0.0044130, -0.0035328, -0.0005289, 0.0004803
4: -0.0002492, 0.0039786, -0.0000747, 0.0041958, -0.0025661, 0.0023301
5: -0.0011137, -0.0004825, -0.0011461, -0.0005086, -0.0003478, 0.0003831
6: 0.9911380, 0.9922956, 0.9911858, 0.9923550, -0.0007026, 0.0006380
7: -0.0138339, -0.0061809, -0.0135181, -0.0057876, -0.0046450, 0.0042179
8: -0.0033457, -0.0009481, -0.0032468, -0.0008249, -0.0014553, 0.0013214
9: -0.0054369, -0.0006516, -0.0056828, -0.0008490, -0.0026374, 0.0029045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010081
time: 1.20 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010081
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058489, 0.0064734, 0.0058814, 0.0065297, -0.0003947, 0.0003343
1: -0.0005892, 0.0006204, -0.0005263, 0.0007295, -0.0007645, 0.0006474
2: 0.0119238, 0.0216800, 0.0110436, 0.0211726, -0.0052222, 0.0061660
3: -0.0044489, -0.0035775, -0.0044036, -0.0034989, -0.0005507, 0.0004664
4: -0.0002492, 0.0039786, -0.0000293, 0.0043600, -0.0026720, 0.0022630
5: -0.0011137, -0.0004825, -0.0011706, -0.0005154, -0.0003378, 0.0003989
6: 0.9911380, 0.9922956, 0.9911982, 0.9924000, -0.0007316, 0.0006196
7: -0.0138339, -0.0061809, -0.0134359, -0.0054905, -0.0048367, 0.0040964
8: -0.0033457, -0.0009481, -0.0032210, -0.0007318, -0.0015153, 0.0012834
9: -0.0054369, -0.0006516, -0.0058686, -0.0009004, -0.0025614, 0.0030244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010168
time: 1.07 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010168
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0058581, 0.0064844, 0.0058786, 0.0064812, -0.0003375, 0.0003462
1: -0.0005712, 0.0006418, -0.0005317, 0.0006356, -0.0006537, 0.0006706
2: 0.0117516, 0.0215355, 0.0118012, 0.0212162, -0.0054090, 0.0052725
3: -0.0044360, -0.0035622, -0.0044075, -0.0035666, -0.0004709, 0.0004831
4: -0.0001866, 0.0040532, -0.0000482, 0.0040317, -0.0022848, 0.0023439
5: -0.0011248, -0.0004919, -0.0011216, -0.0005125, -0.0003499, 0.0003411
6: 0.9911552, 0.9923160, 0.9911931, 0.9923102, -0.0006255, 0.0006417
7: -0.0137205, -0.0060458, -0.0134701, -0.0060848, -0.0041358, 0.0042429
8: -0.0033102, -0.0009058, -0.0032317, -0.0009180, -0.0012957, 0.0013293
9: -0.0055213, -0.0007224, -0.0054970, -0.0008790, -0.0026531, 0.0025861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010077
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010077
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0058581, 0.0064844, 0.0058858, 0.0065066, -0.0003544, 0.0003292
1: -0.0005712, 0.0006418, -0.0005176, 0.0006847, -0.0006865, 0.0006376
2: 0.0117516, 0.0215355, 0.0114049, 0.0211026, -0.0051428, 0.0055373
3: -0.0044360, -0.0035622, -0.0043973, -0.0035312, -0.0004946, 0.0004593
4: -0.0001866, 0.0040532, 0.0000010, 0.0042034, -0.0023995, 0.0022286
5: -0.0011248, -0.0004919, -0.0011472, -0.0005199, -0.0003327, 0.0003582
6: 0.9911552, 0.9923160, 0.9912065, 0.9923571, -0.0006570, 0.0006102
7: -0.0137205, -0.0060458, -0.0133810, -0.0057739, -0.0043436, 0.0040341
8: -0.0033102, -0.0009058, -0.0032038, -0.0008206, -0.0013608, 0.0012638
9: -0.0055213, -0.0007224, -0.0056914, -0.0009348, -0.0025225, 0.0027160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010119
time: 1.21 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010119
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0058581, 0.0064844, 0.0058817, 0.0065053, -0.0003803, 0.0003595
1: -0.0005712, 0.0006418, -0.0005256, 0.0006823, -0.0007366, 0.0006963
2: 0.0117516, 0.0215355, 0.0114245, 0.0211673, -0.0056167, 0.0059411
3: -0.0044360, -0.0035622, -0.0044031, -0.0035329, -0.0005306, 0.0005017
4: -0.0001866, 0.0040532, -0.0000270, 0.0041950, -0.0025745, 0.0024339
5: -0.0011248, -0.0004919, -0.0011460, -0.0005157, -0.0003633, 0.0003843
6: 0.9911552, 0.9923160, 0.9911990, 0.9923549, -0.0007049, 0.0006664
7: -0.0137205, -0.0060458, -0.0134317, -0.0057892, -0.0046603, 0.0044058
8: -0.0033102, -0.0009058, -0.0032197, -0.0008254, -0.0014600, 0.0013803
9: -0.0055213, -0.0007224, -0.0056818, -0.0009030, -0.0027549, 0.0029141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010066
time: 1.24 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010066
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0058581, 0.0064844, 0.0058883, 0.0065296, -0.0003969, 0.0003427
1: -0.0005712, 0.0006418, -0.0005127, 0.0007293, -0.0007689, 0.0006637
2: 0.0117516, 0.0215355, 0.0110458, 0.0210634, -0.0053536, 0.0062015
3: -0.0044360, -0.0035622, -0.0043938, -0.0034991, -0.0005539, 0.0004782
4: -0.0001866, 0.0040532, 0.0000180, 0.0043591, -0.0026874, 0.0023199
5: -0.0011248, -0.0004919, -0.0011705, -0.0005224, -0.0003463, 0.0004012
6: 0.9911552, 0.9923160, 0.9912113, 0.9923998, -0.0007358, 0.0006352
7: -0.0137205, -0.0060458, -0.0133502, -0.0054922, -0.0048646, 0.0041995
8: -0.0033102, -0.0009058, -0.0031942, -0.0007323, -0.0015240, 0.0013157
9: -0.0055213, -0.0007224, -0.0058675, -0.0009540, -0.0026259, 0.0030418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010161
time: 1.36 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010161
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058750, 0.0065019, 0.0058354, 0.0064552, -0.0005802, 0.0003813
1: -0.0005386, 0.0006757, -0.0007068, 0.0005853, -0.0006071, 0.0013825
2: 0.0114777, 0.0212722, 0.0122073, 0.0220057, -0.0105279, 0.0048964
3: -0.0044125, -0.0035377, -0.0044677, -0.0036029, -0.0008096, 0.0005321
4: -0.0000725, 0.0041719, -0.0003405, 0.0038557, -0.0039282, 0.0025816
5: -0.0011425, -0.0005089, -0.0010953, -0.0003940, -0.0007486, 0.0003167
6: 0.9911865, 0.9923485, 0.9909170, 0.9922620, -0.0005809, 0.0014315
7: -0.0135140, -0.0058310, -0.0140371, -0.0064033, -0.0038409, 0.0060930
8: -0.0032455, -0.0008385, -0.0039569, -0.0010178, -0.0012033, 0.0031184
9: -0.0056556, -0.0008516, -0.0052978, -0.0004740, -0.0051817, 0.0024016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008638, upper bound: 0.0008699
time: 1.31 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008229, upper bound: 0.0008499
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058452, 0.0064725, 0.0058354, 0.0064552, -0.0006100, 0.0006371
1: -0.0006197, 0.0006187, -0.0007068, 0.0005853, -0.0012049, 0.0013255
2: 0.0119376, 0.0217662, 0.0122073, 0.0220057, -0.0100680, 0.0095590
3: -0.0044540, -0.0035788, -0.0044677, -0.0036029, -0.0008511, 0.0008890
4: -0.0002738, 0.0039726, -0.0003405, 0.0038557, -0.0041295, 0.0043131
5: -0.0011128, -0.0004597, -0.0010953, -0.0003940, -0.0007188, 0.0006357
6: 0.9910811, 0.9922941, 0.9909170, 0.9922620, -0.0011809, 0.0013771
7: -0.0138881, -0.0061918, -0.0140371, -0.0064033, -0.0053594, 0.0055589
8: -0.0035031, -0.0009515, -0.0039569, -0.0010178, -0.0024853, 0.0030054
9: -0.0054301, -0.0006047, -0.0052978, -0.0004740, -0.0049561, 0.0046931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009901, upper bound: 0.0010377
time: 1.17 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009765, upper bound: 0.0010450
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058750, 0.0065019, 0.0058413, 0.0064752, -0.0006002, 0.0003826
1: -0.0005386, 0.0006757, -0.0006547, 0.0006240, -0.0006443, 0.0013304
2: 0.0114777, 0.0212722, 0.0118945, 0.0218626, -0.0103849, 0.0051969
3: -0.0044125, -0.0035377, -0.0044595, -0.0035749, -0.0008376, 0.0005339
4: -0.0000725, 0.0041719, -0.0003006, 0.0039913, -0.0040637, 0.0025902
5: -0.0011425, -0.0005089, -0.0011155, -0.0004332, -0.0007093, 0.0003362
6: 0.9911865, 0.9923485, 0.9910150, 0.9922991, -0.0006166, 0.0013335
7: -0.0135140, -0.0058310, -0.0139481, -0.0061580, -0.0040766, 0.0060502
8: -0.0032455, -0.0008385, -0.0036857, -0.0009409, -0.0012772, 0.0028473
9: -0.0056556, -0.0008516, -0.0054512, -0.0005521, -0.0051036, 0.0025490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008563, upper bound: 0.0008581
time: 1.11 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008169, upper bound: 0.0008338
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058452, 0.0064725, 0.0058413, 0.0064752, -0.0006300, 0.0006312
1: -0.0006197, 0.0006187, -0.0006547, 0.0006240, -0.0012437, 0.0012734
2: 0.0119376, 0.0217662, 0.0118945, 0.0218626, -0.0099250, 0.0098717
3: -0.0044540, -0.0035788, -0.0044595, -0.0035749, -0.0008791, 0.0008807
4: -0.0002738, 0.0039726, -0.0003006, 0.0039913, -0.0042650, 0.0042732
5: -0.0011128, -0.0004597, -0.0011155, -0.0004332, -0.0006795, 0.0006559
6: 0.9910811, 0.9922941, 0.9910150, 0.9922991, -0.0012180, 0.0012791
7: -0.0138881, -0.0061918, -0.0139481, -0.0061580, -0.0057747, 0.0056592
8: -0.0035031, -0.0009515, -0.0036857, -0.0009409, -0.0025622, 0.0027342
9: -0.0054301, -0.0006047, -0.0054512, -0.0005521, -0.0048780, 0.0048465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010025, upper bound: 0.0010760
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0010025, upper bound: 0.0010760
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0058829, 0.0065142, 0.0058429, 0.0064550, -0.0005721, 0.0004085
1: -0.0005232, 0.0006995, -0.0006406, 0.0005848, -0.0005960, 0.0013401
2: 0.0112861, 0.0211482, 0.0122111, 0.0218238, -0.0105377, 0.0048074
3: -0.0044014, -0.0035206, -0.0044573, -0.0036032, -0.0007982, 0.0005700
4: -0.0000187, 0.0042549, -0.0002898, 0.0038541, -0.0038728, 0.0027655
5: -0.0011549, -0.0005169, -0.0010951, -0.0004439, -0.0007111, 0.0003110
6: 0.9912012, 0.9923713, 0.9910417, 0.9922616, -0.0005704, 0.0013296
7: -0.0134168, -0.0056807, -0.0139239, -0.0064063, -0.0037710, 0.0064088
8: -0.0032150, -0.0007914, -0.0036122, -0.0010187, -0.0011814, 0.0028208
9: -0.0057497, -0.0009124, -0.0052960, -0.0005733, -0.0051764, 0.0023580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008530, upper bound: 0.0008667
time: 1.10 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008108, upper bound: 0.0008468
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0058543, 0.0064831, 0.0058429, 0.0064550, -0.0006007, 0.0003707
1: -0.0005786, 0.0006394, -0.0006406, 0.0005848, -0.0006045, 0.0012800
2: 0.0117708, 0.0215952, 0.0122111, 0.0218238, -0.0100529, 0.0048756
3: -0.0044413, -0.0035639, -0.0044573, -0.0036032, -0.0008381, 0.0005172
4: -0.0002124, 0.0040449, -0.0002898, 0.0038541, -0.0040665, 0.0025094
5: -0.0011236, -0.0004880, -0.0010951, -0.0004439, -0.0006797, 0.0003154
6: 0.9911482, 0.9923137, 0.9910417, 0.9922616, -0.0005785, 0.0012720
7: -0.0137673, -0.0060609, -0.0139239, -0.0064063, -0.0038245, 0.0058864
8: -0.0033249, -0.0009105, -0.0036122, -0.0010187, -0.0011982, 0.0027017
9: -0.0055119, -0.0006932, -0.0052960, -0.0005733, -0.0049386, 0.0023914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008530, upper bound: 0.0009793
time: 1.74 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0008108, upper bound: 0.0009595
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0058829, 0.0065142, 0.0058491, 0.0064751, -0.0003343, 0.0004122
1: -0.0005232, 0.0006995, -0.0005888, 0.0006238, -0.0006476, 0.0007984
2: 0.0112861, 0.0211482, 0.0118965, 0.0216772, -0.0064399, 0.0052232
3: -0.0044014, -0.0035206, -0.0044487, -0.0035751, -0.0004665, 0.0005752
4: -0.0000187, 0.0042549, -0.0002479, 0.0039904, -0.0022634, 0.0027906
5: -0.0011549, -0.0005169, -0.0011154, -0.0004827, -0.0004166, 0.0003379
6: 0.9912012, 0.9923713, 0.9911383, 0.9922988, -0.0006197, 0.0007640
7: -0.0134168, -0.0056807, -0.0138317, -0.0061595, -0.0040972, 0.0050515
8: -0.0032150, -0.0007914, -0.0033450, -0.0009414, -0.0012836, 0.0015826
9: -0.0057497, -0.0009124, -0.0054503, -0.0006529, -0.0031587, 0.0025619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009972, upper bound: 0.0010136
time: 1.18 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0009972, upper bound: 0.0010136
time: 1.20 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_B1_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008825, upper bound: 0.0008542
IS_A1_B1_B1_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008657, upper bound: 0.0008273
IS_A1_B1_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008825, upper bound: 0.0008542
IS_A1_B1_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008657, upper bound: 0.0008273
IS_A1_B1_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008708, upper bound: 0.0008430
IS_A1_B1_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008548, upper bound: 0.0008182
IS_A1_B1_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008708, upper bound: 0.0008430
IS_A1_B1_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008548, upper bound: 0.0008182
IS_A1_B1_B1_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008766, upper bound: 0.0008536
IS_A1_B1_B1_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008579, upper bound: 0.0008264
IS_A1_B1_B1_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009726, upper bound: 0.0009204
IS_A1_B1_B1_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009439, upper bound: 0.0009215
IS_A1_B1_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008658, upper bound: 0.0008425
IS_A1_B1_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008471, upper bound: 0.0008170
IS_A1_B1_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009703, upper bound: 0.0009199
IS_A1_B1_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009414, upper bound: 0.0009204
IS_A1_B1_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010087
IS_A1_B1_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010087
IS_A1_B1_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010129
IS_A1_B1_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010129
IS_A1_B1_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010081
IS_A1_B1_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010081
IS_A1_B1_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010168
IS_A1_B1_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010260, upper bound: 0.0010168
IS_A1_B1_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010077
IS_A1_B1_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010077
IS_A1_B1_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010119
IS_A1_B1_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010208, upper bound: 0.0010119
IS_A1_B1_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010066
IS_A1_B1_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010066
IS_A1_B1_B1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010161
IS_A1_B1_B1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010181, upper bound: 0.0010161
IS_A1_B1_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008638, upper bound: 0.0008699
IS_A1_B1_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008229, upper bound: 0.0008499
IS_A1_B1_B2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009901, upper bound: 0.0010377
IS_A1_B1_B2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009765, upper bound: 0.0010450
IS_A1_B1_B2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008563, upper bound: 0.0008581
IS_A1_B1_B2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008169, upper bound: 0.0008338
IS_A1_B1_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010025, upper bound: 0.0010760
IS_A1_B1_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0010025, upper bound: 0.0010760
IS_A1_B1_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008530, upper bound: 0.0008667
IS_A1_B1_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008108, upper bound: 0.0008468
IS_A1_B1_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008530, upper bound: 0.0009793
IS_A1_B1_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0008108, upper bound: 0.0009595
IS_A1_B1_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009972, upper bound: 0.0010136
IS_A1_B1_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 6, lower bound: -0.0009972, upper bound: 0.0010136
IS_A1_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009982, upper bound: 0.0010758
IS_A1_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010214
IS_A1_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010869
IS_A1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010231
IS_A1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010139, upper bound: 0.0010898
IS_A1_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010213
IS_A1_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010865
IS_A1_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010231
IS_A1_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010098, upper bound: 0.0010894
IS_A1_B2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008508, upper bound: 0.0008727
IS_A1_B2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008236, upper bound: 0.0008540
IS_A1_B2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009243, upper bound: 0.0009729
IS_A1_B2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009256, upper bound: 0.0009449
IS_A1_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008533, upper bound: 0.0008619
IS_A1_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008197, upper bound: 0.0008339
IS_A1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008427, upper bound: 0.0008569
IS_A1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008114, upper bound: 0.0008304
IS_A1_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010258
IS_A1_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010318
IS_A1_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010267
IS_A1_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009986, upper bound: 0.0010344
IS_A1_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010195
IS_A1_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010264
IS_A1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010208
IS_A1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009984, upper bound: 0.0010289
IS_A1_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010011
IS_A1_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010716
IS_A1_B2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010035
IS_A1_B2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0010744
IS_A1_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010006
IS_A1_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010716
IS_A1_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010031
IS_A1_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010183, upper bound: 0.0010744
IS_A1_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010178
IS_A1_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010942
IS_A1_B2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010233
IS_A1_B2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010963
IS_A1_B2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010162
IS_A1_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010912
IS_A1_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010213
IS_A1_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010268, upper bound: 0.0010934
IS_A2_B1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009959
IS_A2_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
IS_A2_B1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
IS_A2_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010237, upper bound: 0.0009960
IS_A2_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009958
IS_A2_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009957
IS_A2_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009958
IS_A2_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010179, upper bound: 0.0009957
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010058
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010058
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010112
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010281, upper bound: 0.0010112
IS_A2_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010048
IS_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010048
IS_A2_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010103
IS_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010219, upper bound: 0.0010103
IS_A2_B1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010137
IS_A2_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010137
IS_A2_B1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010645
IS_A2_B1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010097, upper bound: 0.0010645
IS_A2_B1_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010137
IS_A2_B1_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010137
IS_A2_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010644
IS_A2_B1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010078, upper bound: 0.0010644
IS_A2_B1_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010138
IS_A2_B1_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010231
IS_A2_B1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010760
IS_A2_B1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010070, upper bound: 0.0010898
IS_A2_B1_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010136
IS_A2_B1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010231
IS_A2_B1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010758
IS_A2_B1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010028, upper bound: 0.0010895
IS_A2_B2_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008952, upper bound: 0.0008620
IS_A2_B2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008476, upper bound: 0.0008141
IS_A2_B2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008952, upper bound: 0.0008620
IS_A2_B2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008476, upper bound: 0.0008141
IS_A2_B2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008838, upper bound: 0.0008593
IS_A2_B2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0008349, upper bound: 0.0008111
IS_A2_B2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009691, upper bound: 0.0009249
IS_A2_B2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0009403, upper bound: 0.0009265
IS_A2_B2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0009986
IS_A2_B2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0009986
IS_A2_B2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010151
IS_A2_B2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010222, upper bound: 0.0010151
IS_A2_B2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0009984
IS_A2_B2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0009984
IS_A2_B2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010137
IS_A2_B2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010180, upper bound: 0.0010137
IS_A2_B2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010217
IS_A2_B2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010060
IS_A2_B2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010784
IS_A2_B2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010022, upper bound: 0.0010784
IS_A2_B2_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010031
IS_A2_B2_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010171
IS_A2_B2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010752
IS_A2_B2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010021, upper bound: 0.0010752
IS_A2_B2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010264
IS_A2_B2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010264
IS_A2_B2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010880
IS_A2_B2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010186, upper bound: 0.0010881
IS_A2_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010208
IS_A2_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010289
IS_A2_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010744
IS_A2_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 6, lower bound: -0.0010014, upper bound: 0.0010933

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.69 + 596.44 = 600.13 seconds
