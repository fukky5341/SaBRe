## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.7875192


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813)
1: (-0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893)
2: (-0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501)
3: (0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201)
4: (-0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563)
5: (-0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932)
6: (-0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614)
7: (-0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642)
8: (-0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244)
9: (-0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 2.81 = 4.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.61 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 2.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.84 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 3.84
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
IS_B2, status: Status.UNKNOWN, split count: 1, time: 3.84
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.2592995, 0.2564672, -0.2500582, 0.2443756, -0.5036751, 0.5065254
1: -0.1868174, 0.1974913, -0.1734074, 0.1877671, -0.3745845, 0.3708987
2: -0.1666596, 0.2860169, -0.1552624, 0.2720147, -0.4386743, 0.4412793
3: 0.1520118, 1.0587583, 0.1661023, 1.0579869, -0.9059751, 0.8926560
4: -0.2195624, 0.2034553, -0.2047127, 0.1891258, -0.4086882, 0.4081680
5: -0.1395424, 0.6713460, -0.1254750, 0.6633909, -0.8029333, 0.7968211
6: -0.2119004, 0.2561235, -0.2023995, 0.2440670, -0.4559675, 0.4585229
7: -0.3077715, 0.2163616, -0.2983487, 0.2024147, -0.5101862, 0.5147104
8: -0.2069341, 0.2914328, -0.1971189, 0.2782628, -0.4851969, 0.4885517
9: -0.3397698, 0.3094461, -0.3337545, 0.2865764, -0.6263462, 0.6432005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.82 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.68 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.2635888, 0.2611926, -0.2581915, 0.2553524, -0.5189412, 0.5193841
1: -0.1908717, 0.2009175, -0.1858926, 0.1966751, -0.3875468, 0.3868101
2: -0.1701133, 0.2907368, -0.1658475, 0.2849444, -0.4550577, 0.4565842
3: 0.1429092, 1.0597293, 0.1543276, 1.0586346, -0.9157254, 0.9054018
4: -0.2238975, 0.2077588, -0.2184964, 0.2024755, -0.4263731, 0.4262551
5: -0.1434896, 0.6777036, -0.1385660, 0.6695760, -0.8130656, 0.8162696
6: -0.2152857, 0.2608757, -0.2111069, 0.2549675, -0.4702531, 0.4719827
7: -0.3123749, 0.2203892, -0.3066055, 0.2155051, -0.5278801, 0.5269948
8: -0.2114335, 0.2966909, -0.2057540, 0.2902304, -0.5016639, 0.5024449
9: -0.3438668, 0.3164717, -0.3386965, 0.3081429, -0.6520097, 0.6551682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.76 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.95 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.95
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.2500582, 0.2443756, -0.2500582, 0.2443756, -0.4944339, 0.4944339
1: -0.1734074, 0.1877671, -0.1734074, 0.1877671, -0.3611745, 0.3611745
2: -0.1552624, 0.2720147, -0.1552624, 0.2720147, -0.4272771, 0.4272771
3: 0.1661023, 1.0579869, 0.1661023, 1.0579869, -0.8918846, 0.8918846
4: -0.2047127, 0.1891258, -0.2047127, 0.1891258, -0.3938384, 0.3938384
5: -0.1254750, 0.6633909, -0.1254750, 0.6633909, -0.7888660, 0.7888660
6: -0.2023995, 0.2440670, -0.2023995, 0.2440670, -0.4464665, 0.4464665
7: -0.2983487, 0.2024147, -0.2983487, 0.2024147, -0.5007634, 0.5007634
8: -0.1971189, 0.2782628, -0.1971189, 0.2782628, -0.4753817, 0.4753817
9: -0.3337545, 0.2865764, -0.3337545, 0.2865764, -0.6203309, 0.6203309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8131466, upper bound: 0.8098967
time: 1.53 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8132515, upper bound: 0.8132515
time: 1.50 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.2581915, 0.2553524, -0.2500582, 0.2443756, -0.5025672, 0.5054106
1: -0.1858926, 0.1966751, -0.1734074, 0.1877671, -0.3736597, 0.3700826
2: -0.1658475, 0.2849444, -0.1552624, 0.2720147, -0.4378622, 0.4402067
3: 0.1543276, 1.0586346, 0.1661023, 1.0579869, -0.9036593, 0.8925323
4: -0.2184964, 0.2024755, -0.2047127, 0.1891258, -0.4076221, 0.4071882
5: -0.1385660, 0.6695760, -0.1254750, 0.6633909, -0.8019570, 0.7950511
6: -0.2111069, 0.2549675, -0.2023995, 0.2440670, -0.4551740, 0.4573669
7: -0.3066055, 0.2155051, -0.2983487, 0.2024147, -0.5090203, 0.5138538
8: -0.2057540, 0.2902304, -0.1971189, 0.2782628, -0.4840168, 0.4873493
9: -0.3386965, 0.3081429, -0.3337545, 0.2865764, -0.6252730, 0.6418974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8062971, upper bound: 0.8133581
time: 1.73 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.25 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.2500582, 0.2443756, -0.2581915, 0.2553524, -0.5054106, 0.5025672
1: -0.1734074, 0.1877671, -0.1858926, 0.1966751, -0.3700826, 0.3736597
2: -0.1552624, 0.2720147, -0.1658475, 0.2849444, -0.4402067, 0.4378622
3: 0.1661023, 1.0579869, 0.1543276, 1.0586346, -0.8925323, 0.9036593
4: -0.2047127, 0.1891258, -0.2184964, 0.2024755, -0.4071882, 0.4076221
5: -0.1254750, 0.6633909, -0.1385660, 0.6695760, -0.7950511, 0.8019570
6: -0.2023995, 0.2440670, -0.2111069, 0.2549675, -0.4573669, 0.4551740
7: -0.2983487, 0.2024147, -0.3066055, 0.2155051, -0.5138538, 0.5090203
8: -0.1971189, 0.2782628, -0.2057540, 0.2902304, -0.4873493, 0.4840168
9: -0.3337545, 0.2865764, -0.3386965, 0.3081429, -0.6418974, 0.6252730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
time: 1.64 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.56 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.2581915, 0.2553524, -0.2581915, 0.2553524, -0.5135440, 0.5135440
1: -0.1858926, 0.1966751, -0.1858926, 0.1966751, -0.3825677, 0.3825677
2: -0.1658475, 0.2849444, -0.1658475, 0.2849444, -0.4507918, 0.4507918
3: 0.1543276, 1.0586346, 0.1543276, 1.0586346, -0.9043071, 0.9043071
4: -0.2184964, 0.2024755, -0.2184964, 0.2024755, -0.4209719, 0.4209719
5: -0.1385660, 0.6695760, -0.1385660, 0.6695760, -0.8081421, 0.8081421
6: -0.2111069, 0.2549675, -0.2111069, 0.2549675, -0.4660744, 0.4660744
7: -0.3066055, 0.2155051, -0.3066055, 0.2155051, -0.5221106, 0.5221106
8: -0.2057540, 0.2902304, -0.2057540, 0.2902304, -0.4959844, 0.4959844
9: -0.3386965, 0.3081429, -0.3386965, 0.3081429, -0.6468394, 0.6468394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8098967, upper bound: 0.8131466
time: 1.44 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8132515, upper bound: 0.8132515
time: 1.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.65 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8131466, upper bound: 0.8098967
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8132515, upper bound: 0.8132515
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8062971, upper bound: 0.8133581
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8098967, upper bound: 0.8131466
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 3, lower bound: -0.8132515, upper bound: 0.8132515

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2478594, 0.2419773, -0.1845255, 0.1778992, -0.4257587, 0.4265029
1: -0.1712606, 0.1859410, -0.1148451, 0.1340850, -0.3053456, 0.3007861
2: -0.1534236, 0.2697244, -0.1082478, 0.2020566, -0.3554802, 0.3779722
3: 0.1708458, 1.0576278, 0.3081166, 1.0474061, -0.8765603, 0.7495111
4: -0.2022993, 0.1869364, -0.1320457, 0.1330236, -0.3353228, 0.3189821
5: -0.1232667, 0.6597989, -0.0635694, 0.5561944, -0.6794611, 0.7233682
6: -0.2006754, 0.2416561, -0.1493928, 0.1782129, -0.3788882, 0.3910489
7: -0.2959356, 0.2003753, -0.2311230, 0.1468191, -0.4427547, 0.4314983
8: -0.1947171, 0.2757837, -0.1320903, 0.2037381, -0.3984552, 0.4078740
9: -0.3316899, 0.2830981, -0.2721180, 0.1888764, -0.5205662, 0.5552161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8047298, upper bound: 0.7943315
time: 1.61 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
time: 1.71 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2451406, 0.2390058, -0.2104258, 0.2037751, -0.4489157, 0.4494316
1: -0.1686148, 0.1837253, -0.1368850, 0.1545603, -0.3231751, 0.3206103
2: -0.1511521, 0.2669090, -0.1248386, 0.2310695, -0.3822217, 0.3917476
3: 0.1767793, 1.0570437, 0.2546546, 1.0502827, -0.8735034, 0.8023891
4: -0.1993073, 0.1842445, -0.1610444, 0.1536534, -0.3529606, 0.3452889
5: -0.1205230, 0.6553619, -0.0861511, 0.5965213, -0.7170442, 0.7415130
6: -0.1985678, 0.2386540, -0.1711947, 0.2034858, -0.4020537, 0.4098486
7: -0.2929167, 0.1978709, -0.2558636, 0.1663041, -0.4592208, 0.4537345
8: -0.1916909, 0.2726879, -0.1556225, 0.2333108, -0.4250017, 0.4283104
9: -0.3290939, 0.2786594, -0.2952935, 0.2264405, -0.5555345, 0.5739529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8051654, upper bound: 0.7984227
time: 1.33 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.57 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2581915, 0.2553524, -0.1928342, 0.1904607, -0.4486522, 0.4481866
1: -0.1858926, 0.1966751, -0.1316828, 0.1445950, -0.3304876, 0.3283579
2: -0.1658475, 0.2849444, -0.1210749, 0.2188999, -0.3847474, 0.4060193
3: 0.1543276, 1.0586346, 0.2998670, 1.0480038, -0.8936762, 0.7587676
4: -0.2184964, 0.2024755, -0.1570106, 0.1462089, -0.3647052, 0.3594862
5: -0.1385660, 0.6695760, -0.0832145, 0.5591776, -0.6977437, 0.7527906
6: -0.2111069, 0.2549675, -0.1616564, 0.1887369, -0.3998438, 0.4166239
7: -0.3066055, 0.2155051, -0.2378954, 0.1618498, -0.4684553, 0.4534005
8: -0.2057540, 0.2902304, -0.1405334, 0.2173305, -0.4230845, 0.4307638
9: -0.3386965, 0.3081429, -0.2729709, 0.2185826, -0.5572791, 0.5811138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982349, upper bound: 0.8019043
time: 1.77 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
time: 1.50 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2411986, 0.2385994, -0.1700392, 0.1697491, -0.4109477, 0.4086386
1: -0.1729589, 0.1840058, -0.1168623, 0.1270175, -0.2999765, 0.3008681
2: -0.1549075, 0.2693508, -0.1095840, 0.1973391, -0.3522466, 0.3789348
3: 0.1934627, 1.0555109, 0.3535259, 1.0441525, -0.8506898, 0.7019850
4: -0.2044517, 0.1887232, -0.1376660, 0.1310732, -0.3355249, 0.3263892
5: -0.1259177, 0.6388491, -0.0681767, 0.5157401, -0.6416578, 0.7070258
6: -0.1992192, 0.2376846, -0.1447474, 0.1678536, -0.3670728, 0.3824320
7: -0.2880291, 0.2035665, -0.2137316, 0.1502388, -0.4382679, 0.4172981
8: -0.1880914, 0.2722728, -0.1186267, 0.1933837, -0.3814750, 0.3908995
9: -0.3208368, 0.2871346, -0.2485444, 0.1931556, -0.5139924, 0.5356790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
time: 1.38 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.50 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1928342, 0.1904607, -0.2581915, 0.2553524, -0.4481866, 0.4486522
1: -0.1316828, 0.1445950, -0.1858926, 0.1966751, -0.3283579, 0.3304876
2: -0.1210749, 0.2188999, -0.1658475, 0.2849444, -0.4060193, 0.3847474
3: 0.2998670, 1.0480038, 0.1543276, 1.0586346, -0.7587676, 0.8936762
4: -0.1570106, 0.1462089, -0.2184964, 0.2024755, -0.3594862, 0.3647052
5: -0.0832145, 0.5591776, -0.1385660, 0.6695760, -0.7527906, 0.6977437
6: -0.1616564, 0.1887369, -0.2111069, 0.2549675, -0.4166239, 0.3998438
7: -0.2378954, 0.1618498, -0.3066055, 0.2155051, -0.4534005, 0.4684553
8: -0.1405334, 0.2173305, -0.2057540, 0.2902304, -0.4307638, 0.4230845
9: -0.2729709, 0.2185826, -0.3386965, 0.3081429, -0.5811138, 0.5572791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8019043, upper bound: 0.7982349
time: 1.31 seconds

## Relational analysis of IS_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8052873, upper bound: 0.7984227
time: 1.93 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.1700392, 0.1697491, -0.2411986, 0.2385994, -0.4086386, 0.4109477
1: -0.1168623, 0.1270175, -0.1729589, 0.1840058, -0.3008681, 0.2999765
2: -0.1095840, 0.1973391, -0.1549075, 0.2693508, -0.3789348, 0.3522466
3: 0.3535259, 1.0441525, 0.1934627, 1.0555109, -0.7019850, 0.8506898
4: -0.1376660, 0.1310732, -0.2044517, 0.1887232, -0.3263892, 0.3355249
5: -0.0681767, 0.5157401, -0.1259177, 0.6388491, -0.7070258, 0.6416578
6: -0.1447474, 0.1678536, -0.1992192, 0.2376846, -0.3824320, 0.3670728
7: -0.2137316, 0.1502388, -0.2880291, 0.2035665, -0.4172981, 0.4382679
8: -0.1186267, 0.1933837, -0.1880914, 0.2722728, -0.3908995, 0.3814750
9: -0.2485444, 0.1931556, -0.3208368, 0.2871346, -0.5356790, 0.5139924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 1.27 seconds

## Relational analysis of IS_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.92 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1944888, 0.1894062, -0.2560593, 0.2530384, -0.4475272, 0.4454656
1: -0.1268509, 0.1424298, -0.1838257, 0.1948888, -0.3217396, 0.3262556
2: -0.1170127, 0.2164593, -0.1640802, 0.2827053, -0.3997180, 0.3805395
3: 0.2917957, 1.0477788, 0.1588859, 1.0582733, -0.7664776, 0.8888929
4: -0.1479239, 0.1433751, -0.2161759, 0.2003401, -0.3482640, 0.3595509
5: -0.0755049, 0.5656476, -0.1364474, 0.6661286, -0.7416335, 0.7020950
6: -0.1596299, 0.1889379, -0.2094397, 0.2526312, -0.4122611, 0.3983776
7: -0.2387442, 0.1578449, -0.3042561, 0.2135400, -0.4522841, 0.4621010
8: -0.1400375, 0.2166153, -0.2034210, 0.2878103, -0.4278478, 0.4200364
9: -0.2774511, 0.2108361, -0.3366863, 0.3048103, -0.5822614, 0.5475224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7943315, upper bound: 0.8048641
time: 1.48 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 1.70 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.2135898, 0.2080126, -0.2528463, 0.2495341, -0.4631240, 0.4608589
1: -0.1434188, 0.1585454, -0.1807194, 0.1922438, -0.3356627, 0.3392648
2: -0.1298020, 0.2378683, -0.1614109, 0.2793531, -0.4091551, 0.3992792
3: 0.2531691, 1.0498550, 0.1659386, 1.0575924, -0.8044233, 0.8839164
4: -0.1690800, 0.1592627, -0.2126716, 0.1971309, -0.3662110, 0.3719343
5: -0.0935079, 0.5949740, -0.1332400, 0.6609286, -0.7544365, 0.7282140
6: -0.1756191, 0.2073455, -0.2069426, 0.2491031, -0.4247222, 0.4142881
7: -0.2572718, 0.1747610, -0.3006642, 0.2106123, -0.4678841, 0.4754252
8: -0.1572704, 0.2388131, -0.1998404, 0.2841473, -0.4414176, 0.4386534
9: -0.2951636, 0.2383493, -0.3336403, 0.2996078, -0.5947714, 0.5719896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
time: 2.43 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.31 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.8047298, upper bound: 0.7943315
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.8051654, upper bound: 0.7984227
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7982349, upper bound: 0.8019043
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.8019043, upper bound: 0.7982349
IS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.8052873, upper bound: 0.7984227
IS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7943315, upper bound: 0.8048641
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.31
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1907662, 0.1883931, -0.1845255, 0.1778992, -0.3686654, 0.3729187
1: -0.1297893, 0.1428752, -0.1148451, 0.1340850, -0.2638743, 0.2577203
2: -0.1196004, 0.2166394, -0.1082478, 0.2020566, -0.3216570, 0.3248872
3: 0.3043905, 1.0476724, 0.3081166, 1.0474061, -0.7430155, 0.7395557
4: -0.1546790, 0.1444633, -0.1320457, 0.1330236, -0.2877026, 0.2765090
5: -0.0811333, 0.5558370, -0.0635694, 0.5561944, -0.6373277, 0.6194064
6: -0.1599279, 0.1867008, -0.1493928, 0.1782129, -0.3381408, 0.3360936
7: -0.2358180, 0.1598139, -0.2311230, 0.1468191, -0.3826371, 0.3909370
8: -0.1385832, 0.2148885, -0.1320903, 0.2037381, -0.3423213, 0.3469787
9: -0.2709928, 0.2154495, -0.2721180, 0.1888764, -0.4598692, 0.4875675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
time: 1.20 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
time: 1.47 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1679339, 0.1676483, -0.1692703, 0.1638452, -0.3317791, 0.3369187
1: -0.1151401, 0.1252743, -0.1049854, 0.1258158, -0.2409559, 0.2302597
2: -0.1082044, 0.1950246, -0.1024585, 0.1874259, -0.2956302, 0.2974831
3: 0.3580763, 1.0437949, 0.3421411, 1.0447450, -0.6866686, 0.7016538
4: -0.1353173, 0.1294409, -0.1207210, 0.1230654, -0.2583827, 0.2501619
5: -0.0663766, 0.5123629, -0.0559390, 0.5288242, -0.5952008, 0.5683019
6: -0.1429878, 0.1658226, -0.1387420, 0.1641185, -0.3071063, 0.3045645
7: -0.2116149, 0.1488235, -0.2167451, 0.1389361, -0.3505510, 0.3655686
8: -0.1166318, 0.1909144, -0.1177679, 0.1888807, -0.3055125, 0.3086822
9: -0.2466342, 0.1900074, -0.2563769, 0.1739528, -0.4205870, 0.4463843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
time: 1.51 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915029
time: 1.62 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1881291, 0.1857213, -0.2104258, 0.2037751, -0.3919042, 0.3961471
1: -0.1275861, 0.1406989, -0.1368850, 0.1545603, -0.2821464, 0.2775838
2: -0.1178365, 0.2137821, -0.1248386, 0.2310695, -0.3489061, 0.3386207
3: 0.3102655, 1.0470703, 0.2546546, 1.0502827, -0.7400172, 0.7924157
4: -0.1516979, 0.1424364, -0.1610444, 0.1536534, -0.3053513, 0.3034808
5: -0.0787834, 0.5514922, -0.0861511, 0.5965213, -0.6753047, 0.6376433
6: -0.1577414, 0.1841331, -0.1711947, 0.2034858, -0.3612272, 0.3553277
7: -0.2330974, 0.1578968, -0.2558636, 0.1663041, -0.3994015, 0.4137604
8: -0.1359937, 0.2117562, -0.1556225, 0.2333108, -0.3693045, 0.3673787
9: -0.2684828, 0.2113369, -0.2952935, 0.2264405, -0.4949234, 0.5066304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.49 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.26 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1653704, 0.1650427, -0.1952326, 0.1896932, -0.3550636, 0.3602753
1: -0.1130243, 0.1231347, -0.1264413, 0.1431483, -0.2561726, 0.2495760
2: -0.1064946, 0.1922304, -0.1166476, 0.2164965, -0.3229911, 0.3088780
3: 0.3637943, 1.0432264, 0.2902004, 1.0476573, -0.6838629, 0.7530260
4: -0.1324185, 0.1274688, -0.1479938, 0.1433285, -0.2757469, 0.2754626
5: -0.0641453, 0.5082189, -0.0754920, 0.5692375, -0.6333829, 0.5837109
6: -0.1408453, 0.1633405, -0.1599323, 0.1893151, -0.3301604, 0.3232729
7: -0.2089848, 0.1470835, -0.2403015, 0.1566465, -0.3656313, 0.3873850
8: -0.1141128, 0.1878678, -0.1413366, 0.2167391, -0.3308519, 0.3292044
9: -0.2442836, 0.1860731, -0.2794225, 0.2084175, -0.4527011, 0.4654956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
time: 1.34 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7954968
time: 1.32 seconds

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2560593, 0.2530384, -0.1317191, 0.1289298, -0.3849891, 0.3847575
1: -0.1838257, 0.1948888, -0.0848706, 0.1041383, -0.2879640, 0.2797593
2: -0.1640802, 0.2827053, -0.0872983, 0.1543125, -0.3183928, 0.3700036
3: 0.1588859, 1.0582733, 0.4291662, 1.0377334, -0.8788475, 0.6291071
4: -0.2161759, 0.2003401, -0.0970853, 0.1024454, -0.3186213, 0.2974254
5: -0.1364474, 0.6661286, -0.0452760, 0.4561366, -0.5925840, 0.7114047
6: -0.2094397, 0.2526312, -0.1132267, 0.1291009, -0.3385406, 0.3658578
7: -0.3042561, 0.2135400, -0.1814547, 0.1186827, -0.4229388, 0.3949947
8: -0.2034210, 0.2878103, -0.0854347, 0.1524939, -0.3559150, 0.3732450
9: -0.3366863, 0.3048103, -0.2148649, 0.1535578, -0.4902441, 0.5196751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982349, upper bound: 0.8019043
time: 1.51 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982349, upper bound: 0.8019043
time: 1.84 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2528463, 0.2495341, -0.1542416, 0.1519016, -0.4047479, 0.4037758
1: -0.1807194, 0.1922438, -0.1005289, 0.1175174, -0.2982369, 0.2927727
2: -0.1614109, 0.2793531, -0.0990607, 0.1775641, -0.3389750, 0.3784138
3: 0.1659386, 1.0575924, 0.3843683, 1.0402501, -0.8743114, 0.6732241
4: -0.2126716, 0.1971309, -0.1166608, 0.1170698, -0.3297414, 0.3137918
5: -0.1332400, 0.6609286, -0.0537671, 0.4945304, -0.6277704, 0.7146958
6: -0.2069426, 0.2491031, -0.1309222, 0.1514606, -0.3584032, 0.3800253
7: -0.3006642, 0.2106123, -0.1998606, 0.1358128, -0.4364770, 0.4104729
8: -0.1998404, 0.2841473, -0.1027434, 0.1752706, -0.3751110, 0.3868906
9: -0.3336403, 0.2996078, -0.2365111, 0.1664075, -0.5000477, 0.5361190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B1_A2_B1_B2_A1

### Relational analysis result of IS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
time: 1.56 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2

### Relational analysis result of IS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
time: 1.64 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2390556, 0.2362715, -0.1105519, 0.1101886, -0.3492442, 0.3468234
1: -0.1708833, 0.1822171, -0.0740228, 0.0910839, -0.2619673, 0.2562398
2: -0.1531318, 0.2670971, -0.0785275, 0.1382448, -0.2913766, 0.3456246
3: 0.1980511, 1.0551447, 0.4790385, 1.0330607, -0.8350096, 0.5761061
4: -0.2021225, 0.1865669, -0.0841818, 0.0904121, -0.2925346, 0.2707487
5: -0.1237888, 0.6353976, -0.0392090, 0.4127526, -0.5365415, 0.6746066
6: -0.1975401, 0.2353354, -0.0983005, 0.1092908, -0.3068309, 0.3336360
7: -0.2856619, 0.2015952, -0.1604647, 0.1075805, -0.3932424, 0.3620600
8: -0.1857444, 0.2698289, -0.0676629, 0.1315599, -0.3173043, 0.3374918
9: -0.3188177, 0.2837821, -0.1898751, 0.1449287, -0.4637464, 0.4736571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
time: 1.20 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
time: 2.29 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2358579, 0.2327744, -0.1330035, 0.1334874, -0.3693453, 0.3657779
1: -0.1677929, 0.1795749, -0.0893807, 0.1049781, -0.2727711, 0.2689556
2: -0.1504795, 0.2637631, -0.0907634, 0.1587194, -0.3091989, 0.3545265
3: 0.2050315, 1.0544637, 0.4325308, 1.0361400, -0.8311085, 0.6219329
4: -0.1986355, 0.1833571, -0.1034804, 0.1055048, -0.3041403, 0.2868375
5: -0.1206035, 0.6302522, -0.0478313, 0.4524018, -0.5730052, 0.6780836
6: -0.1950424, 0.2318229, -0.1167040, 0.1321764, -0.3272188, 0.3485268
7: -0.2820945, 0.1986873, -0.1796581, 0.1254147, -0.4075092, 0.3783453
8: -0.1821752, 0.2661716, -0.0841531, 0.1558038, -0.3379791, 0.3503247
9: -0.3158064, 0.2786133, -0.2128599, 0.1569544, -0.4727609, 0.4914732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.63 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.47 seconds

## BFS IS instance: IS_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1317191, 0.1289298, -0.2560593, 0.2530384, -0.3847575, 0.3849891
1: -0.0848706, 0.1041383, -0.1838257, 0.1948888, -0.2797593, 0.2879640
2: -0.0872983, 0.1543125, -0.1640802, 0.2827053, -0.3700036, 0.3183928
3: 0.4291662, 1.0377334, 0.1588859, 1.0582733, -0.6291071, 0.8788475
4: -0.0970853, 0.1024454, -0.2161759, 0.2003401, -0.2974254, 0.3186213
5: -0.0452760, 0.4561366, -0.1364474, 0.6661286, -0.7114047, 0.5925840
6: -0.1132267, 0.1291009, -0.2094397, 0.2526312, -0.3658578, 0.3385406
7: -0.1814547, 0.1186827, -0.3042561, 0.2135400, -0.3949947, 0.4229388
8: -0.0854347, 0.1524939, -0.2034210, 0.2878103, -0.3732450, 0.3559150
9: -0.2148649, 0.1535578, -0.3366863, 0.3048103, -0.5196751, 0.4902441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8019043, upper bound: 0.7982349
time: 1.70 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8019043, upper bound: 0.7982349
time: 1.24 seconds

## BFS IS instance: IS_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1542416, 0.1519016, -0.2528463, 0.2495341, -0.4037758, 0.4047479
1: -0.1005289, 0.1175174, -0.1807194, 0.1922438, -0.2927727, 0.2982369
2: -0.0990607, 0.1775641, -0.1614109, 0.2793531, -0.3784138, 0.3389750
3: 0.3843683, 1.0402501, 0.1659386, 1.0575924, -0.6732241, 0.8743114
4: -0.1166608, 0.1170698, -0.2126716, 0.1971309, -0.3137918, 0.3297414
5: -0.0537671, 0.4945304, -0.1332400, 0.6609286, -0.7146958, 0.6277704
6: -0.1309222, 0.1514606, -0.2069426, 0.2491031, -0.3800253, 0.3584032
7: -0.1998606, 0.1358128, -0.3006642, 0.2106123, -0.4104729, 0.4364770
8: -0.1027434, 0.1752706, -0.1998404, 0.2841473, -0.3868906, 0.3751110
9: -0.2365111, 0.1664075, -0.3336403, 0.2996078, -0.5361190, 0.5000477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B2_A1_A1_A2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8052873, upper bound: 0.7984227
time: 2.92 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2

### Relational analysis result of IS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8052873, upper bound: 0.7984227
time: 1.47 seconds

## BFS IS instance: IS_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1105519, 0.1101886, -0.2390556, 0.2362715, -0.3468234, 0.3492442
1: -0.0740228, 0.0910839, -0.1708833, 0.1822171, -0.2562398, 0.2619673
2: -0.0785275, 0.1382448, -0.1531318, 0.2670971, -0.3456246, 0.2913766
3: 0.4790385, 1.0330607, 0.1980511, 1.0551447, -0.5761061, 0.8350096
4: -0.0841818, 0.0904121, -0.2021225, 0.1865669, -0.2707487, 0.2925346
5: -0.0392090, 0.4127526, -0.1237888, 0.6353976, -0.6746066, 0.5365415
6: -0.0983005, 0.1092908, -0.1975401, 0.2353354, -0.3336360, 0.3068309
7: -0.1604647, 0.1075805, -0.2856619, 0.2015952, -0.3620600, 0.3932424
8: -0.0676629, 0.1315599, -0.1857444, 0.2698289, -0.3374918, 0.3173043
9: -0.1898751, 0.1449287, -0.3188177, 0.2837821, -0.4736571, 0.4637464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 1.55 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 1.28 seconds

## BFS IS instance: IS_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.1330035, 0.1334874, -0.2358579, 0.2327744, -0.3657779, 0.3693453
1: -0.0893807, 0.1049781, -0.1677929, 0.1795749, -0.2689556, 0.2727711
2: -0.0907634, 0.1587194, -0.1504795, 0.2637631, -0.3545265, 0.3091989
3: 0.4325308, 1.0361400, 0.2050315, 1.0544637, -0.6219329, 0.8311085
4: -0.1034804, 0.1055048, -0.1986355, 0.1833571, -0.2868375, 0.3041403
5: -0.0478313, 0.4524018, -0.1206035, 0.6302522, -0.6780836, 0.5730052
6: -0.1167040, 0.1321764, -0.1950424, 0.2318229, -0.3485268, 0.3272188
7: -0.1796581, 0.1254147, -0.2820945, 0.1986873, -0.3783453, 0.4075092
8: -0.0841531, 0.1558038, -0.1821752, 0.2661716, -0.3503247, 0.3379791
9: -0.2128599, 0.1569544, -0.3158064, 0.2786133, -0.4914732, 0.4727609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.36 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.37 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1944888, 0.1894062, -0.2008129, 0.1991272, -0.3936160, 0.3902191
1: -0.1268509, 0.1424298, -0.1414057, 0.1525677, -0.2794186, 0.2838355
2: -0.1170127, 0.2164593, -0.1284882, 0.2305561, -0.3475688, 0.3449475
3: 0.2917957, 1.0477788, 0.2872586, 1.0479835, -0.7561879, 0.7605203
4: -0.1479239, 0.1433751, -0.1691514, 0.1554652, -0.3033891, 0.3125264
5: -0.0755049, 0.5656476, -0.0940534, 0.5662068, -0.6417116, 0.6597010
6: -0.1596299, 0.1889379, -0.1697285, 0.1970201, -0.3566500, 0.3586664
7: -0.2387442, 0.1578449, -0.2442824, 0.1735987, -0.4123428, 0.4021273
8: -0.1400375, 0.2166153, -0.1467438, 0.2279482, -0.3679858, 0.3633592
9: -0.2774511, 0.2108361, -0.2779949, 0.2363034, -0.5137545, 0.4888310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 1.67 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 1.24 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1781097, 0.1743776, -0.1677723, 0.1718529, -0.3499626, 0.3421499
1: -0.1162822, 0.1302456, -0.1198221, 0.1275419, -0.2438241, 0.2500677
2: -0.1087124, 0.2007699, -0.1118888, 0.2002357, -0.3089481, 0.3126586
3: 0.3297426, 1.0447334, 0.3658765, 1.0412790, -0.7115364, 0.6788570
4: -0.1342047, 0.1326185, -0.1430477, 0.1334732, -0.2676780, 0.2756662
5: -0.0650880, 0.5363216, -0.0723758, 0.5031993, -0.5682873, 0.6086974
6: -0.1475807, 0.1738734, -0.1461396, 0.1669474, -0.3145280, 0.3200130
7: -0.2219701, 0.1494163, -0.2093948, 0.1536616, -0.3756317, 0.3588110
8: -0.1246589, 0.1989008, -0.1154788, 0.1932937, -0.3179526, 0.3143796
9: -0.2605451, 0.1917381, -0.2415068, 0.2006663, -0.4612114, 0.4332450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 1.54 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
time: 2.66 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2135898, 0.2080126, -0.1976253, 0.1960113, -0.4096012, 0.4056379
1: -0.1434188, 0.1585454, -0.1385711, 0.1498151, -0.2932339, 0.2971165
2: -0.1298020, 0.2378683, -0.1262705, 0.2270650, -0.3568670, 0.3641388
3: 0.2531691, 1.0498550, 0.2942019, 1.0472732, -0.7941041, 0.7556531
4: -0.1690800, 0.1592627, -0.1655132, 0.1528008, -0.3218808, 0.3247759
5: -0.0935079, 0.5949740, -0.0909062, 0.5611748, -0.6546827, 0.6858802
6: -0.1756191, 0.2073455, -0.1671342, 0.1939621, -0.3695812, 0.3744797
7: -0.2572718, 0.1747610, -0.2410232, 0.1705802, -0.4278520, 0.4157841
8: -0.1572704, 0.2388131, -0.1436184, 0.2242301, -0.3815005, 0.3824314
9: -0.2951636, 0.2383493, -0.2749504, 0.2314071, -0.5265707, 0.5132997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.33 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1972475, 0.1932019, -0.1648141, 0.1689845, -0.3662320, 0.3580160
1: -0.1319943, 0.1461250, -0.1174115, 0.1250812, -0.2570755, 0.2635365
2: -0.1209654, 0.2223045, -0.1099447, 0.1969857, -0.3179511, 0.3322492
3: 0.2908697, 1.0467600, 0.3723415, 1.0406098, -0.7497401, 0.6744185
4: -0.1553480, 0.1478350, -0.1397577, 0.1311591, -0.2865071, 0.2875926
5: -0.0816050, 0.5657979, -0.0698306, 0.4985538, -0.5801588, 0.6356286
6: -0.1637251, 0.1922733, -0.1436874, 0.1641219, -0.3278469, 0.3359607
7: -0.2405221, 0.1630007, -0.2063839, 0.1516864, -0.3922085, 0.3693846
8: -0.1419812, 0.2211456, -0.1125969, 0.1898096, -0.3317908, 0.3337424
9: -0.2777822, 0.2192472, -0.2388351, 0.1960995, -0.4738817, 0.4580823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.44 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
time: 1.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.42 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915029
IS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7954968
IS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982349, upper bound: 0.8019043
IS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982349, upper bound: 0.8019043
IS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
IS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7984227, upper bound: 0.8052873
IS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
IS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7979191, upper bound: 0.7940984
IS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.8019043, upper bound: 0.7982349
IS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.8019043, upper bound: 0.7982349
IS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.8052873, upper bound: 0.7984227
IS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.8052873, upper bound: 0.7984227
IS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7940984, upper bound: 0.7979191
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -0.7982071, upper bound: 0.7982071

## BFS IS instance: IS_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1907662, 0.1883931, -0.1317191, 0.1289298, -0.3196959, 0.3201123
1: -0.1297893, 0.1428752, -0.0848706, 0.1041383, -0.2339277, 0.2277457
2: -0.1196004, 0.2166394, -0.0872983, 0.1543125, -0.2739129, 0.3039377
3: 0.3043905, 1.0476724, 0.4291662, 1.0377334, -0.7333429, 0.6185062
4: -0.1546790, 0.1444633, -0.0970853, 0.1024454, -0.2571245, 0.2415486
5: -0.0811333, 0.5558370, -0.0452760, 0.4561366, -0.5372699, 0.6011131
6: -0.1599279, 0.1867008, -0.1132267, 0.1291009, -0.2890288, 0.2999275
7: -0.2358180, 0.1598139, -0.1814547, 0.1186827, -0.3545007, 0.3412687
8: -0.1385832, 0.2148885, -0.0854347, 0.1524939, -0.2910771, 0.3003232
9: -0.2709928, 0.2154495, -0.2148649, 0.1535578, -0.4245507, 0.4303144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7940029
time: 1.89 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917203
time: 2.21 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1907662, 0.1883931, -0.1105519, 0.1101886, -0.3009548, 0.2989450
1: -0.1297893, 0.1428752, -0.0740228, 0.0910839, -0.2208733, 0.2168979
2: -0.1196004, 0.2166394, -0.0785275, 0.1382448, -0.2578452, 0.2951669
3: 0.3043905, 1.0476724, 0.4790385, 1.0330607, -0.7286701, 0.5686339
4: -0.1546790, 0.1444633, -0.0841818, 0.0904121, -0.2450911, 0.2286451
5: -0.0811333, 0.5558370, -0.0392090, 0.4127526, -0.4938859, 0.5950460
6: -0.1599279, 0.1867008, -0.0983005, 0.1092908, -0.2692187, 0.2850013
7: -0.2358180, 0.1598139, -0.1604647, 0.1075805, -0.3433985, 0.3202787
8: -0.1385832, 0.2148885, -0.0676629, 0.1315599, -0.2701431, 0.2825513
9: -0.2709928, 0.2154495, -0.1898751, 0.1449287, -0.4159215, 0.4053246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7940029
time: 1.97 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917203
time: 1.68 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1567600, 0.1568212, -0.1683654, 0.1629759, -0.3197359, 0.3251866
1: -0.1064691, 0.1189440, -0.1043390, 0.1253662, -0.2318353, 0.2232831
2: -0.1024963, 0.1829841, -0.1020608, 0.1864913, -0.2889876, 0.2850449
3: 0.3797373, 1.0415432, 0.3438785, 1.0445536, -0.6648163, 0.6976647
4: -0.1248855, 0.1212094, -0.1199480, 0.1224672, -0.2473528, 0.2411575
5: -0.0589422, 0.4962245, -0.0554630, 0.5274475, -0.5863897, 0.5516875
6: -0.1345023, 0.1552241, -0.1380950, 0.1632548, -0.2977571, 0.2933191
7: -0.2018844, 0.1412248, -0.2159687, 0.1383536, -0.3402381, 0.3571936
8: -0.1062965, 0.1790221, -0.1169269, 0.1879722, -0.2942688, 0.2959490
9: -0.2366939, 0.1756240, -0.2555208, 0.1729169, -0.4096108, 0.4311448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
time: 1.38 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
time: 1.78 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.1808042, 0.1795978, -0.1684202, 0.1630156, -0.3438198, 0.3480179
1: -0.1240668, 0.1353924, -0.1043562, 0.1253809, -0.2494478, 0.2397486
2: -0.1151550, 0.2080397, -0.1020760, 0.1865289, -0.3016839, 0.3101157
3: 0.3288566, 1.0454627, 0.3437884, 1.0445851, -0.7157285, 0.7016743
4: -0.1472030, 0.1383668, -0.1199611, 0.1224844, -0.2696874, 0.2583279
5: -0.0753658, 0.5354205, -0.0554718, 0.5275155, -0.6028813, 0.5908923
6: -0.1529576, 0.1779296, -0.1381181, 0.1632987, -0.3162562, 0.3160477
7: -0.2247660, 0.1558320, -0.2160146, 0.1383761, -0.3631421, 0.3718466
8: -0.1285102, 0.2051155, -0.1169789, 0.1880269, -0.3165371, 0.3220945
9: -0.2597371, 0.2049792, -0.2555847, 0.1729695, -0.4327067, 0.4605640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915029
time: 1.52 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915029
time: 1.79 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1881291, 0.1857213, -0.1542416, 0.1519016, -0.3400307, 0.3399629
1: -0.1275861, 0.1406989, -0.1005289, 0.1175174, -0.2451036, 0.2412278
2: -0.1178365, 0.2137821, -0.0990607, 0.1775641, -0.2954006, 0.3128427
3: 0.3102655, 1.0470703, 0.3843683, 1.0402501, -0.7299845, 0.6627020
4: -0.1516979, 0.1424364, -0.1166608, 0.1170698, -0.2687677, 0.2590972
5: -0.0787834, 0.5514922, -0.0537671, 0.4945304, -0.5733139, 0.6052594
6: -0.1577414, 0.1841331, -0.1309222, 0.1514606, -0.3092020, 0.3150553
7: -0.2330974, 0.1578968, -0.1998606, 0.1358128, -0.3689103, 0.3577574
8: -0.1359937, 0.2117562, -0.1027434, 0.1752706, -0.3112642, 0.3144996
9: -0.2684828, 0.2113369, -0.2365111, 0.1664075, -0.4348903, 0.4478480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023643, upper bound: 0.7980576
time: 1.66 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957172
time: 2.00 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1881291, 0.1857213, -0.1330035, 0.1334874, -0.3216165, 0.3187248
1: -0.1275861, 0.1406989, -0.0893807, 0.1049781, -0.2325643, 0.2300796
2: -0.1178365, 0.2137821, -0.0907634, 0.1587194, -0.2765559, 0.3045454
3: 0.3102655, 1.0470703, 0.4325308, 1.0361400, -0.7258744, 0.6145394
4: -0.1516979, 0.1424364, -0.1034804, 0.1055048, -0.2572027, 0.2459168
5: -0.0787834, 0.5514922, -0.0478313, 0.4524018, -0.5311852, 0.5993236
6: -0.1577414, 0.1841331, -0.1167040, 0.1321764, -0.2899178, 0.3008370
7: -0.2330974, 0.1578968, -0.1796581, 0.1254147, -0.3585121, 0.3375548
8: -0.1359937, 0.2117562, -0.0841531, 0.1558038, -0.2917975, 0.2959093
9: -0.2684828, 0.2113369, -0.2128599, 0.1569544, -0.4254372, 0.4241968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023643, upper bound: 0.7980576
time: 1.34 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957172
time: 1.39 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1541927, 0.1542200, -0.1943075, 0.1888168, -0.3430095, 0.3485276
1: -0.1043523, 0.1175890, -0.1257643, 0.1424771, -0.2468295, 0.2433533
2: -0.1012686, 0.1801942, -0.1160977, 0.2155431, -0.3168118, 0.2962919
3: 0.3851002, 1.0409769, 0.2920780, 1.0474596, -0.6623594, 0.7488989
4: -0.1223764, 0.1192424, -0.1471146, 0.1426849, -0.2650613, 0.2663569
5: -0.0572329, 0.4920993, -0.0748137, 0.5678605, -0.6250935, 0.5669130
6: -0.1325160, 0.1527448, -0.1592237, 0.1884484, -0.3209644, 0.3119685
7: -0.1995640, 0.1394854, -0.2394238, 0.1560530, -0.3556170, 0.3789092
8: -0.1037776, 0.1764105, -0.1404895, 0.2156761, -0.3194537, 0.3169000
9: -0.2343406, 0.1723778, -0.2785648, 0.2071416, -0.4414822, 0.4509426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
time: 1.59 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
time: 1.65 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.1780732, 0.1768743, -0.1943357, 0.1888304, -0.3669036, 0.3712100
1: -0.1218534, 0.1331627, -0.1257584, 0.1424667, -0.2643200, 0.2589211
2: -0.1133669, 0.2050979, -0.1160953, 0.2155464, -0.3289133, 0.3211932
3: 0.3349604, 1.0448638, 0.2920325, 1.0474887, -0.7125282, 0.7528313
4: -0.1441875, 0.1362880, -0.1470899, 0.1426782, -0.2868656, 0.2833779
5: -0.0730292, 0.5310109, -0.0747956, 0.5678933, -0.6409224, 0.6058065
6: -0.1507127, 0.1753033, -0.1592225, 0.1884658, -0.3391785, 0.3345258
7: -0.2219921, 0.1540061, -0.2394434, 0.1560574, -0.3780495, 0.3934495
8: -0.1258658, 0.2019123, -0.1405176, 0.2157013, -0.3415672, 0.3424299
9: -0.2572277, 0.2007945, -0.2786052, 0.2071444, -0.4643722, 0.4793997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B2_A2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7952164
time: 1.45 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7954565
time: 1.37 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2008129, 0.1991272, -0.1317191, 0.1289298, -0.3297427, 0.3308464
1: -0.1414057, 0.1525677, -0.0848706, 0.1041383, -0.2455440, 0.2374383
2: -0.1284882, 0.2305561, -0.0872983, 0.1543125, -0.2828008, 0.3178543
3: 0.2872586, 1.0479835, 0.4291662, 1.0377334, -0.7504749, 0.6188173
4: -0.1691514, 0.1554652, -0.0970853, 0.1024454, -0.2715968, 0.2525505
5: -0.0940534, 0.5662068, -0.0452760, 0.4561366, -0.5501901, 0.6114828
6: -0.1697285, 0.1970201, -0.1132267, 0.1291009, -0.2988294, 0.3102468
7: -0.2442824, 0.1735987, -0.1814547, 0.1186827, -0.3629650, 0.3550534
8: -0.1467438, 0.2279482, -0.0854347, 0.1524939, -0.2992377, 0.3133829
9: -0.2779949, 0.2363034, -0.2148649, 0.1535578, -0.4315527, 0.4511682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978170, upper bound: 0.7990156
time: 1.33 seconds

## Relational analysis of IS_B1_A2_B1_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955412, upper bound: 0.7990125
time: 1.50 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1677723, 0.1718529, -0.1317191, 0.1289298, -0.2967021, 0.3035721
1: -0.1198221, 0.1275419, -0.0848706, 0.1041383, -0.2239604, 0.2124125
2: -0.1118888, 0.2002357, -0.0872983, 0.1543125, -0.2662013, 0.2875340
3: 0.3658765, 1.0412790, 0.4291662, 1.0377334, -0.6718570, 0.6121128
4: -0.1430477, 0.1334732, -0.0970853, 0.1024454, -0.2454931, 0.2305585
5: -0.0723758, 0.5031993, -0.0452760, 0.4561366, -0.5285124, 0.5484753
6: -0.1461396, 0.1669474, -0.1132267, 0.1291009, -0.2752405, 0.2801740
7: -0.2093948, 0.1536616, -0.1814547, 0.1186827, -0.3280774, 0.3351163
8: -0.1154788, 0.1932937, -0.0854347, 0.1524939, -0.2679728, 0.2787284
9: -0.2415068, 0.2006663, -0.2148649, 0.1535578, -0.3950647, 0.4155311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978170, upper bound: 0.7990156
time: 2.51 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955412, upper bound: 0.7990125
time: 2.76 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1976253, 0.1960113, -0.1542416, 0.1519016, -0.3495269, 0.3502529
1: -0.1385711, 0.1498151, -0.1005289, 0.1175174, -0.2560885, 0.2503440
2: -0.1262705, 0.2270650, -0.0990607, 0.1775641, -0.3038346, 0.3261256
3: 0.2942019, 1.0472732, 0.3843683, 1.0402501, -0.7460482, 0.6629049
4: -0.1655132, 0.1528008, -0.1166608, 0.1170698, -0.2825830, 0.2694616
5: -0.0909062, 0.5611748, -0.0537671, 0.4945304, -0.5854366, 0.6149420
6: -0.1671342, 0.1939621, -0.1309222, 0.1514606, -0.3185948, 0.3248843
7: -0.2410232, 0.1705802, -0.1998606, 0.1358128, -0.3768360, 0.3704408
8: -0.1436184, 0.2242301, -0.1027434, 0.1752706, -0.3188890, 0.3269735
9: -0.2749504, 0.2314071, -0.2365111, 0.1664075, -0.4413579, 0.4679183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B2_A1_A1

### Relational analysis result of IS_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
time: 1.83 seconds

## Relational analysis of IS_B1_A2_B1_B2_A1_A2

### Relational analysis result of IS_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
time: 1.31 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1648141, 0.1689845, -0.1542416, 0.1519016, -0.3167157, 0.3232262
1: -0.1174115, 0.1250812, -0.1005289, 0.1175174, -0.2349290, 0.2256101
2: -0.1099447, 0.1969857, -0.0990607, 0.1775641, -0.2875088, 0.2960464
3: 0.3723415, 1.0406098, 0.3843683, 1.0402501, -0.6679086, 0.6562415
4: -0.1397577, 0.1311591, -0.1166608, 0.1170698, -0.2568275, 0.2478199
5: -0.0698306, 0.4985538, -0.0537671, 0.4945304, -0.5643611, 0.5523210
6: -0.1436874, 0.1641219, -0.1309222, 0.1514606, -0.2951480, 0.2950441
7: -0.2063839, 0.1516864, -0.1998606, 0.1358128, -0.3421968, 0.3515471
8: -0.1125969, 0.1898096, -0.1027434, 0.1752706, -0.2878675, 0.2925529
9: -0.2388351, 0.1960995, -0.2365111, 0.1664075, -0.4052426, 0.4326106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
time: 1.58 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
time: 1.91 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2008129, 0.1991272, -0.1105519, 0.1101886, -0.3110015, 0.3096791
1: -0.1414057, 0.1525677, -0.0740228, 0.0910839, -0.2324896, 0.2265905
2: -0.1284882, 0.2305561, -0.0785275, 0.1382448, -0.2667331, 0.3090836
3: 0.2872586, 1.0479835, 0.4790385, 1.0330607, -0.7458021, 0.5689450
4: -0.1691514, 0.1554652, -0.0841818, 0.0904121, -0.2595634, 0.2396470
5: -0.0940534, 0.5662068, -0.0392090, 0.4127526, -0.5068061, 0.6054158
6: -0.1697285, 0.1970201, -0.0983005, 0.1092908, -0.2790193, 0.2953206
7: -0.2442824, 0.1735987, -0.1604647, 0.1075805, -0.3518629, 0.3340634
8: -0.1467438, 0.2279482, -0.0676629, 0.1315599, -0.2783037, 0.2956111
9: -0.2779949, 0.2363034, -0.1898751, 0.1449287, -0.4229236, 0.4261785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952745, upper bound: 0.7938348
time: 1.57 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
time: 2.10 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1677723, 0.1718529, -0.1105519, 0.1101886, -0.2779609, 0.2824048
1: -0.1198221, 0.1275419, -0.0740228, 0.0910839, -0.2109060, 0.2015646
2: -0.1118888, 0.2002357, -0.0785275, 0.1382448, -0.2501336, 0.2787632
3: 0.3658765, 1.0412790, 0.4790385, 1.0330607, -0.6671842, 0.5622404
4: -0.1430477, 0.1334732, -0.0841818, 0.0904121, -0.2334598, 0.2176550
5: -0.0723758, 0.5031993, -0.0392090, 0.4127526, -0.4851284, 0.5424083
6: -0.1461396, 0.1669474, -0.0983005, 0.1092908, -0.2554305, 0.2652479
7: -0.2093948, 0.1536616, -0.1604647, 0.1075805, -0.3169752, 0.3141264
8: -0.1154788, 0.1932937, -0.0676629, 0.1315599, -0.2470387, 0.2609566
9: -0.2415068, 0.2006663, -0.1898751, 0.1449287, -0.3864355, 0.3905414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
time: 1.27 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
time: 1.32 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1976253, 0.1960113, -0.1330035, 0.1334874, -0.3311127, 0.3290148
1: -0.1385711, 0.1498151, -0.0893807, 0.1049781, -0.2435492, 0.2391959
2: -0.1262705, 0.2270650, -0.0907634, 0.1587194, -0.2849899, 0.3178283
3: 0.2942019, 1.0472732, 0.4325308, 1.0361400, -0.7419381, 0.6147423
4: -0.1655132, 0.1528008, -0.1034804, 0.1055048, -0.2710180, 0.2562812
5: -0.0909062, 0.5611748, -0.0478313, 0.4524018, -0.5433080, 0.6090062
6: -0.1671342, 0.1939621, -0.1167040, 0.1321764, -0.2993106, 0.3106660
7: -0.2410232, 0.1705802, -0.1796581, 0.1254147, -0.3664378, 0.3502382
8: -0.1436184, 0.2242301, -0.0841531, 0.1558038, -0.2994222, 0.3083832
9: -0.2749504, 0.2314071, -0.2128599, 0.1569544, -0.4319048, 0.4442670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955306, upper bound: 0.7979120
time: 1.32 seconds

## Relational analysis of IS_B1_A2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000
time: 1.43 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1648141, 0.1689845, -0.1330035, 0.1334874, -0.2983015, 0.3019880
1: -0.1174115, 0.1250812, -0.0893807, 0.1049781, -0.2223897, 0.2144619
2: -0.1099447, 0.1969857, -0.0907634, 0.1587194, -0.2686640, 0.2877491
3: 0.3723415, 1.0406098, 0.4325308, 1.0361400, -0.6637985, 0.6080790
4: -0.1397577, 0.1311591, -0.1034804, 0.1055048, -0.2452625, 0.2346395
5: -0.0698306, 0.4985538, -0.0478313, 0.4524018, -0.5222324, 0.5463852
6: -0.1436874, 0.1641219, -0.1167040, 0.1321764, -0.2758638, 0.2808258
7: -0.2063839, 0.1516864, -0.1796581, 0.1254147, -0.3317986, 0.3313445
8: -0.1125969, 0.1898096, -0.0841531, 0.1558038, -0.2684007, 0.2739627
9: -0.2388351, 0.1960995, -0.2128599, 0.1569544, -0.3957895, 0.4089594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
time: 1.61 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000
time: 1.46 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1317191, 0.1289298, -0.2008129, 0.1991272, -0.3308464, 0.3297427
1: -0.0848706, 0.1041383, -0.1414057, 0.1525677, -0.2374383, 0.2455440
2: -0.0872983, 0.1543125, -0.1284882, 0.2305561, -0.3178543, 0.2828008
3: 0.4291662, 1.0377334, 0.2872586, 1.0479835, -0.6188173, 0.7504749
4: -0.0970853, 0.1024454, -0.1691514, 0.1554652, -0.2525505, 0.2715968
5: -0.0452760, 0.4561366, -0.0940534, 0.5662068, -0.6114828, 0.5501901
6: -0.1132267, 0.1291009, -0.1697285, 0.1970201, -0.3102468, 0.2988294
7: -0.1814547, 0.1186827, -0.2442824, 0.1735987, -0.3550534, 0.3629650
8: -0.0854347, 0.1524939, -0.1467438, 0.2279482, -0.3133829, 0.2992377
9: -0.2148649, 0.1535578, -0.2779949, 0.2363034, -0.4511682, 0.4315527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7990156, upper bound: 0.7978170
time: 1.67 seconds

## Relational analysis of IS_B2_A1_A1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7990125, upper bound: 0.7955412
time: 1.46 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1317191, 0.1289298, -0.1677723, 0.1718529, -0.3035721, 0.2967021
1: -0.0848706, 0.1041383, -0.1198221, 0.1275419, -0.2124125, 0.2239604
2: -0.0872983, 0.1543125, -0.1118888, 0.2002357, -0.2875340, 0.2662013
3: 0.4291662, 1.0377334, 0.3658765, 1.0412790, -0.6121128, 0.6718570
4: -0.0970853, 0.1024454, -0.1430477, 0.1334732, -0.2305585, 0.2454931
5: -0.0452760, 0.4561366, -0.0723758, 0.5031993, -0.5484753, 0.5285124
6: -0.1132267, 0.1291009, -0.1461396, 0.1669474, -0.2801740, 0.2752405
7: -0.1814547, 0.1186827, -0.2093948, 0.1536616, -0.3351163, 0.3280774
8: -0.0854347, 0.1524939, -0.1154788, 0.1932937, -0.2787284, 0.2679728
9: -0.2148649, 0.1535578, -0.2415068, 0.2006663, -0.4155311, 0.3950647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7990156, upper bound: 0.7978170
time: 1.48 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2_B2

### Relational analysis result of IS_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7990125, upper bound: 0.7955412
time: 1.51 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1542416, 0.1519016, -0.1976253, 0.1960113, -0.3502529, 0.3495269
1: -0.1005289, 0.1175174, -0.1385711, 0.1498151, -0.2503440, 0.2560885
2: -0.0990607, 0.1775641, -0.1262705, 0.2270650, -0.3261256, 0.3038346
3: 0.3843683, 1.0402501, 0.2942019, 1.0472732, -0.6629049, 0.7460482
4: -0.1166608, 0.1170698, -0.1655132, 0.1528008, -0.2694616, 0.2825830
5: -0.0537671, 0.4945304, -0.0909062, 0.5611748, -0.6149420, 0.5854366
6: -0.1309222, 0.1514606, -0.1671342, 0.1939621, -0.3248843, 0.3185948
7: -0.1998606, 0.1358128, -0.2410232, 0.1705802, -0.3704408, 0.3768360
8: -0.1027434, 0.1752706, -0.1436184, 0.2242301, -0.3269735, 0.3188890
9: -0.2365111, 0.1664075, -0.2749504, 0.2314071, -0.4679183, 0.4413579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8024712, upper bound: 0.7980576
time: 1.52 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8024688, upper bound: 0.7957172
time: 2.59 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1542416, 0.1519016, -0.1648141, 0.1689845, -0.3232262, 0.3167157
1: -0.1005289, 0.1175174, -0.1174115, 0.1250812, -0.2256101, 0.2349290
2: -0.0990607, 0.1775641, -0.1099447, 0.1969857, -0.2960464, 0.2875088
3: 0.3843683, 1.0402501, 0.3723415, 1.0406098, -0.6562415, 0.6679086
4: -0.1166608, 0.1170698, -0.1397577, 0.1311591, -0.2478199, 0.2568275
5: -0.0537671, 0.4945304, -0.0698306, 0.4985538, -0.5523210, 0.5643611
6: -0.1309222, 0.1514606, -0.1436874, 0.1641219, -0.2950441, 0.2951480
7: -0.1998606, 0.1358128, -0.2063839, 0.1516864, -0.3515471, 0.3421968
8: -0.1027434, 0.1752706, -0.1125969, 0.1898096, -0.2925529, 0.2878675
9: -0.2365111, 0.1664075, -0.2388351, 0.1960995, -0.4326106, 0.4052426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8024712, upper bound: 0.7980576
time: 2.21 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_B2

### Relational analysis result of IS_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8024688, upper bound: 0.7957172
time: 1.67 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1105519, 0.1101886, -0.2008129, 0.1991272, -0.3096791, 0.3110015
1: -0.0740228, 0.0910839, -0.1414057, 0.1525677, -0.2265905, 0.2324896
2: -0.0785275, 0.1382448, -0.1284882, 0.2305561, -0.3090836, 0.2667331
3: 0.4790385, 1.0330607, 0.2872586, 1.0479835, -0.5689450, 0.7458021
4: -0.0841818, 0.0904121, -0.1691514, 0.1554652, -0.2396470, 0.2595634
5: -0.0392090, 0.4127526, -0.0940534, 0.5662068, -0.6054158, 0.5068061
6: -0.0983005, 0.1092908, -0.1697285, 0.1970201, -0.2953206, 0.2790193
7: -0.1604647, 0.1075805, -0.2442824, 0.1735987, -0.3340634, 0.3518629
8: -0.0676629, 0.1315599, -0.1467438, 0.2279482, -0.2956111, 0.2783037
9: -0.1898751, 0.1449287, -0.2779949, 0.2363034, -0.4261785, 0.4229236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A1_B1_A1

### Relational analysis result of IS_B2_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7938348, upper bound: 0.7952745
time: 1.36 seconds

## Relational analysis of IS_B2_A1_A2_A1_B1_A2

### Relational analysis result of IS_B2_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
time: 1.40 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1105519, 0.1101886, -0.1677723, 0.1718529, -0.2824048, 0.2779609
1: -0.0740228, 0.0910839, -0.1198221, 0.1275419, -0.2015646, 0.2109060
2: -0.0785275, 0.1382448, -0.1118888, 0.2002357, -0.2787632, 0.2501336
3: 0.4790385, 1.0330607, 0.3658765, 1.0412790, -0.5622404, 0.6671842
4: -0.0841818, 0.0904121, -0.1430477, 0.1334732, -0.2176550, 0.2334598
5: -0.0392090, 0.4127526, -0.0723758, 0.5031993, -0.5424083, 0.4851284
6: -0.0983005, 0.1092908, -0.1461396, 0.1669474, -0.2652479, 0.2554305
7: -0.1604647, 0.1075805, -0.2093948, 0.1536616, -0.3141264, 0.3169752
8: -0.0676629, 0.1315599, -0.1154788, 0.1932937, -0.2609566, 0.2470387
9: -0.1898751, 0.1449287, -0.2415068, 0.2006663, -0.3905414, 0.3864355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A1_B2_B1

### Relational analysis result of IS_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915430, upper bound: 0.7974645
time: 1.51 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
time: 2.20 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1330035, 0.1334874, -0.1976253, 0.1960113, -0.3290148, 0.3311127
1: -0.0893807, 0.1049781, -0.1385711, 0.1498151, -0.2391959, 0.2435492
2: -0.0907634, 0.1587194, -0.1262705, 0.2270650, -0.3178283, 0.2849899
3: 0.4325308, 1.0361400, 0.2942019, 1.0472732, -0.6147423, 0.7419381
4: -0.1034804, 0.1055048, -0.1655132, 0.1528008, -0.2562812, 0.2710180
5: -0.0478313, 0.4524018, -0.0909062, 0.5611748, -0.6090062, 0.5433080
6: -0.1167040, 0.1321764, -0.1671342, 0.1939621, -0.3106660, 0.2993106
7: -0.1796581, 0.1254147, -0.2410232, 0.1705802, -0.3502382, 0.3664378
8: -0.0841531, 0.1558038, -0.1436184, 0.2242301, -0.3083832, 0.2994222
9: -0.2128599, 0.1569544, -0.2749504, 0.2314071, -0.4442670, 0.4319048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979120, upper bound: 0.7955306
time: 1.54 seconds

## Relational analysis of IS_B2_A1_A2_A2_B1_A2

### Relational analysis result of IS_B2_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955000, upper bound: 0.7954968
time: 1.97 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1330035, 0.1334874, -0.1648141, 0.1689845, -0.3019880, 0.2983015
1: -0.0893807, 0.1049781, -0.1174115, 0.1250812, -0.2144619, 0.2223897
2: -0.0907634, 0.1587194, -0.1099447, 0.1969857, -0.2877491, 0.2686640
3: 0.4325308, 1.0361400, 0.3723415, 1.0406098, -0.6080790, 0.6637985
4: -0.1034804, 0.1055048, -0.1397577, 0.1311591, -0.2346395, 0.2452625
5: -0.0478313, 0.4524018, -0.0698306, 0.4985538, -0.5463852, 0.5222324
6: -0.1167040, 0.1321764, -0.1436874, 0.1641219, -0.2808258, 0.2758638
7: -0.1796581, 0.1254147, -0.2063839, 0.1516864, -0.3313445, 0.3317986
8: -0.0841531, 0.1558038, -0.1125969, 0.1898096, -0.2739627, 0.2684007
9: -0.2128599, 0.1569544, -0.2388351, 0.1960995, -0.4089594, 0.3957895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A2_B2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955306, upper bound: 0.7978070
time: 1.65 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2_B2

### Relational analysis result of IS_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955000, upper bound: 0.7954968
time: 2.16 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1417397, 0.1408900, -0.2008129, 0.1991272, -0.3408670, 0.3417028
1: -0.0941959, 0.1102504, -0.1414057, 0.1525677, -0.2467636, 0.2516561
2: -0.0944369, 0.1663948, -0.1284882, 0.2305561, -0.3249930, 0.2948830
3: 0.4110149, 1.0377470, 0.2872586, 1.0479835, -0.6369687, 0.7504885
4: -0.1085244, 0.1103841, -0.1691514, 0.1554652, -0.2639896, 0.2795355
5: -0.0501218, 0.4700379, -0.0940534, 0.5662068, -0.6163285, 0.5640913
6: -0.1226668, 0.1401622, -0.1697285, 0.1970201, -0.3196869, 0.3098907
7: -0.1878023, 0.1301928, -0.2442824, 0.1735987, -0.3614010, 0.3744752
8: -0.0913325, 0.1638477, -0.1467438, 0.2279482, -0.3192807, 0.3105915
9: -0.2222893, 0.1618313, -0.2779949, 0.2363034, -0.4585927, 0.4398262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940029, upper bound: 0.8020389
time: 1.78 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917203, upper bound: 0.8020113
time: 1.37 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1115575, 0.1156875, -0.2008129, 0.1991272, -0.3106847, 0.3165004
1: -0.0776112, 0.0912379, -0.1414057, 0.1525677, -0.2301789, 0.2326436
2: -0.0815555, 0.1397822, -0.1284882, 0.2305561, -0.3121116, 0.2682705
3: 0.4853680, 1.0306849, 0.2872586, 1.0479835, -0.5626155, 0.7434264
4: -0.0896991, 0.0933632, -0.1691514, 0.1554652, -0.2451643, 0.2625146
5: -0.0413810, 0.4067812, -0.0940534, 0.5662068, -0.6075878, 0.5008346
6: -0.1017215, 0.1108666, -0.1697285, 0.1970201, -0.2987416, 0.2805951
7: -0.1570608, 0.1136805, -0.2442824, 0.1735987, -0.3306594, 0.3579629
8: -0.0645078, 0.1377207, -0.1467438, 0.2279482, -0.2924560, 0.2844644
9: -0.1886409, 0.1475149, -0.2779949, 0.2363034, -0.4249443, 0.4255098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7940029, upper bound: 0.8020389
time: 1.59 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917203, upper bound: 0.8020113
time: 1.48 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1417397, 0.1408900, -0.1677723, 0.1718529, -0.3135927, 0.3086622
1: -0.0941959, 0.1102504, -0.1198221, 0.1275419, -0.2217378, 0.2300725
2: -0.0944369, 0.1663948, -0.1118888, 0.2002357, -0.2946726, 0.2782835
3: 0.4110149, 1.0377470, 0.3658765, 1.0412790, -0.6302641, 0.6718706
4: -0.1085244, 0.1103841, -0.1430477, 0.1334732, -0.2419976, 0.2534318
5: -0.0501218, 0.4700379, -0.0723758, 0.5031993, -0.5533210, 0.5424137
6: -0.1226668, 0.1401622, -0.1461396, 0.1669474, -0.2896142, 0.2863019
7: -0.1878023, 0.1301928, -0.2093948, 0.1536616, -0.3414639, 0.3395876
8: -0.0913325, 0.1638477, -0.1154788, 0.1932937, -0.2846262, 0.2793266
9: -0.2222893, 0.1618313, -0.2415068, 0.2006663, -0.4229556, 0.4033381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915430, upper bound: 0.7975857
time: 1.59 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915029, upper bound: 0.7952321
time: 1.51 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1115575, 0.1156875, -0.1677723, 0.1718529, -0.2834104, 0.2834598
1: -0.0776112, 0.0912379, -0.1198221, 0.1275419, -0.2051531, 0.2110600
2: -0.0815555, 0.1397822, -0.1118888, 0.2002357, -0.2817912, 0.2516710
3: 0.4853680, 1.0306849, 0.3658765, 1.0412790, -0.5559109, 0.6648085
4: -0.0896991, 0.0933632, -0.1430477, 0.1334732, -0.2231724, 0.2364109
5: -0.0413810, 0.4067812, -0.0723758, 0.5031993, -0.5445803, 0.4791570
6: -0.1017215, 0.1108666, -0.1461396, 0.1669474, -0.2686689, 0.2570062
7: -0.1570608, 0.1136805, -0.2093948, 0.1536616, -0.3107224, 0.3230752
8: -0.0645078, 0.1377207, -0.1154788, 0.1932937, -0.2578015, 0.2531995
9: -0.1886409, 0.1475149, -0.2415068, 0.2006663, -0.3893072, 0.3890218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937535, upper bound: 0.7952745
time: 1.43 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915029, upper bound: 0.7952321
time: 2.19 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1588797, 0.1579672, -0.1976253, 0.1960113, -0.3548910, 0.3555924
1: -0.1066560, 0.1199113, -0.1385711, 0.1498151, -0.2564712, 0.2584824
2: -0.1026314, 0.1851690, -0.1262705, 0.2270650, -0.3296964, 0.3114395
3: 0.3782677, 1.0394742, 0.2942019, 1.0472732, -0.6690055, 0.7452724
4: -0.1240542, 0.1220796, -0.1655132, 0.1528008, -0.2768550, 0.2875928
5: -0.0581378, 0.4976993, -0.0909062, 0.5611748, -0.6193126, 0.5886055
6: -0.1357999, 0.1569666, -0.1671342, 0.1939621, -0.3297620, 0.3241007
7: -0.2022960, 0.1421639, -0.2410232, 0.1705802, -0.3728762, 0.3831871
8: -0.1060200, 0.1809222, -0.1436184, 0.2242301, -0.3302501, 0.3245406
9: -0.2384084, 0.1760652, -0.2749504, 0.2314071, -0.4698156, 0.4510156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
time: 1.42 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
time: 1.50 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1283130, 0.1338618, -0.1976253, 0.1960113, -0.3243244, 0.3314871
1: -0.0897638, 0.1021778, -0.1385711, 0.1498151, -0.2395789, 0.2407489
2: -0.0909009, 0.1575274, -0.1262705, 0.2270650, -0.3179659, 0.2837979
3: 0.4489268, 1.0328033, 0.2942019, 1.0472732, -0.5983464, 0.7386014
4: -0.1047856, 0.1048700, -0.1655132, 0.1528008, -0.2575864, 0.2703833
5: -0.0479240, 0.4377977, -0.0909062, 0.5611748, -0.6090989, 0.5287039
6: -0.1156079, 0.1289840, -0.1671342, 0.1939621, -0.3095700, 0.2961182
7: -0.1723208, 0.1273861, -0.2410232, 0.1705802, -0.3429010, 0.3684092
8: -0.0782046, 0.1531372, -0.1436184, 0.2242301, -0.3024348, 0.2967556
9: -0.2046262, 0.1569735, -0.2749504, 0.2314071, -0.4360334, 0.4319239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
time: 1.44 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
time: 1.41 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1588797, 0.1579672, -0.1648141, 0.1689845, -0.3278642, 0.3227813
1: -0.1066560, 0.1199113, -0.1174115, 0.1250812, -0.2317372, 0.2373228
2: -0.1026314, 0.1851690, -0.1099447, 0.1969857, -0.2996171, 0.2951137
3: 0.3782677, 1.0394742, 0.3723415, 1.0406098, -0.6623421, 0.6671328
4: -0.1240542, 0.1220796, -0.1397577, 0.1311591, -0.2552133, 0.2618372
5: -0.0581378, 0.4976993, -0.0698306, 0.4985538, -0.5566916, 0.5675299
6: -0.1357999, 0.1569666, -0.1436874, 0.1641219, -0.2999218, 0.3006539
7: -0.2022960, 0.1421639, -0.2063839, 0.1516864, -0.3539825, 0.3485478
8: -0.1060200, 0.1809222, -0.1125969, 0.1898096, -0.2958296, 0.2935191
9: -0.2384084, 0.1760652, -0.2388351, 0.1960995, -0.4345079, 0.4149003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955306, upper bound: 0.7979120
time: 1.51 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000
time: 1.44 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1283130, 0.1338618, -0.1648141, 0.1689845, -0.2972975, 0.2986760
1: -0.0897638, 0.1021778, -0.1174115, 0.1250812, -0.2148450, 0.2195893
2: -0.0909009, 0.1575274, -0.1099447, 0.1969857, -0.2878866, 0.2674721
3: 0.4489268, 1.0328033, 0.3723415, 1.0406098, -0.5916830, 0.6604618
4: -0.1047856, 0.1048700, -0.1397577, 0.1311591, -0.2359447, 0.2446277
5: -0.0479240, 0.4377977, -0.0698306, 0.4985538, -0.5464779, 0.5076284
6: -0.1156079, 0.1289840, -0.1436874, 0.1641219, -0.2797298, 0.2726714
7: -0.1723208, 0.1273861, -0.2063839, 0.1516864, -0.3240073, 0.3337700
8: -0.0782046, 0.1531372, -0.1125969, 0.1898096, -0.2680142, 0.2657341
9: -0.2046262, 0.1569735, -0.2388351, 0.1960995, -0.4007257, 0.3958086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
time: 1.71 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000
time: 1.50 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.71 seconds
IS_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7940029
IS_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917203
IS_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7940029
IS_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917203
IS_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
IS_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
IS_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915029
IS_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915029
IS_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8023643, upper bound: 0.7980576
IS_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957172
IS_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8023643, upper bound: 0.7980576
IS_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957172
IS_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
IS_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
IS_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7952164
IS_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7954565
IS_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7978170, upper bound: 0.7990156
IS_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7955412, upper bound: 0.7990125
IS_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7978170, upper bound: 0.7990156
IS_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7955412, upper bound: 0.7990125
IS_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
IS_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
IS_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
IS_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
IS_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7952745, upper bound: 0.7938348
IS_B1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
IS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
IS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
IS_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7955306, upper bound: 0.7979120
IS_B1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000
IS_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
IS_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000
IS_B2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7990156, upper bound: 0.7978170
IS_B2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7990125, upper bound: 0.7955412
IS_B2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7990156, upper bound: 0.7978170
IS_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7990125, upper bound: 0.7955412
IS_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8024712, upper bound: 0.7980576
IS_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8024688, upper bound: 0.7957172
IS_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8024712, upper bound: 0.7980576
IS_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.8024688, upper bound: 0.7957172
IS_B2_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7938348, upper bound: 0.7952745
IS_B2_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
IS_B2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7915430, upper bound: 0.7974645
IS_B2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
IS_B2_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7979120, upper bound: 0.7955306
IS_B2_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7955000, upper bound: 0.7954968
IS_B2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7955306, upper bound: 0.7978070
IS_B2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7955000, upper bound: 0.7954968
IS_B2_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7940029, upper bound: 0.8020389
IS_B2_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7917203, upper bound: 0.8020113
IS_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7940029, upper bound: 0.8020389
IS_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7917203, upper bound: 0.8020113
IS_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7915430, upper bound: 0.7975857
IS_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7915029, upper bound: 0.7952321
IS_B2_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7937535, upper bound: 0.7952745
IS_B2_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7915029, upper bound: 0.7952321
IS_B2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
IS_B2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
IS_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7980576, upper bound: 0.8024712
IS_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7957172, upper bound: 0.8024688
IS_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7955306, upper bound: 0.7979120
IS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000
IS_B2_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
IS_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.71
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000

## BFS IS instance: IS_B1_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1898155, 0.1874935, -0.1226750, 0.1195307, -0.3093463, 0.3101686
1: -0.1290555, 0.1421886, -0.0791085, 0.0989467, -0.2280022, 0.2212971
2: -0.1190164, 0.2156549, -0.0824709, 0.1467876, -0.2658040, 0.2981258
3: 0.3063102, 1.0474737, 0.4479043, 1.0356437, -0.7293335, 0.5995693
4: -0.1537762, 0.1437683, -0.0900757, 0.0966551, -0.2504313, 0.2338440
5: -0.0803824, 0.5544282, -0.0420813, 0.4405155, -0.5208979, 0.5965095
6: -0.1591997, 0.1858049, -0.1061381, 0.1200152, -0.2792149, 0.2919430
7: -0.2349224, 0.1590791, -0.1738738, 0.1115120, -0.3464344, 0.3329529
8: -0.1377136, 0.2137944, -0.0786737, 0.1425952, -0.2803088, 0.2924682
9: -0.2700978, 0.2141165, -0.2054584, 0.1479486, -0.4180464, 0.4195749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
time: 2.36 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
time: 1.57 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1898876, 0.1875503, -0.1486047, 0.1457115, -0.3355991, 0.3361550
1: -0.1290875, 0.1422131, -0.0941434, 0.1148630, -0.2439505, 0.2363565
2: -0.1190444, 0.2157087, -0.0947579, 0.1675950, -0.2866393, 0.3104666
3: 0.3061816, 1.0475097, 0.3878068, 1.0404202, -0.7342386, 0.6597028
4: -0.1538031, 0.1437976, -0.1082674, 0.1126018, -0.2664050, 0.2520650
5: -0.0804040, 0.5545161, -0.0503279, 0.4914005, -0.5718045, 0.6048440
6: -0.1592354, 0.1858649, -0.1258655, 0.1455541, -0.3047895, 0.3117304
7: -0.2349823, 0.1591161, -0.1979781, 0.1286469, -0.3636292, 0.3570942
8: -0.1377804, 0.2138717, -0.0991993, 0.1699160, -0.3076965, 0.3130710
9: -0.2701719, 0.2141893, -0.2342879, 0.1611518, -0.4313238, 0.4484772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
time: 1.34 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
time: 2.41 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1898155, 0.1874935, -0.1023928, 0.1014286, -0.2912441, 0.2898864
1: -0.1290555, 0.1421886, -0.0683402, 0.0861951, -0.2152506, 0.2105288
2: -0.1190164, 0.2156549, -0.0738268, 0.1308653, -0.2498817, 0.2894817
3: 0.3063102, 1.0474737, 0.4973913, 1.0313350, -0.7250248, 0.5500824
4: -0.1537762, 0.1437683, -0.0771136, 0.0863025, -0.2400787, 0.2208819
5: -0.0803824, 0.5544282, -0.0360820, 0.3978105, -0.4781929, 0.5905101
6: -0.1591997, 0.1858049, -0.0920873, 0.1003963, -0.2595960, 0.2778922
7: -0.2349224, 0.1590791, -0.1533401, 0.1005147, -0.3354371, 0.3124192
8: -0.1377136, 0.2137944, -0.0615834, 0.1234170, -0.2611306, 0.2753778
9: -0.2700978, 0.2141165, -0.1819123, 0.1395789, -0.4096767, 0.3960288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7939819
time: 1.37 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7940029
time: 1.83 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1898876, 0.1875503, -0.1227538, 0.1210608, -0.3109483, 0.3103041
1: -0.1290875, 0.1422131, -0.0809997, 0.0986160, -0.2277035, 0.2232127
2: -0.1190444, 0.2157087, -0.0840840, 0.1479863, -0.2670307, 0.2997926
3: 0.3061816, 1.0475097, 0.4501366, 1.0350879, -0.7289063, 0.5973730
4: -0.1538031, 0.1437976, -0.0927137, 0.0963777, -0.2501808, 0.2365113
5: -0.0804040, 0.5545161, -0.0431789, 0.4368120, -0.5172160, 0.5976950
6: -0.1592354, 0.1858649, -0.1068448, 0.1213654, -0.2806008, 0.2927098
7: -0.2349823, 0.1591161, -0.1719452, 0.1151237, -0.3501061, 0.3310613
8: -0.1377804, 0.2138717, -0.0769005, 0.1443600, -0.2821404, 0.2907722
9: -0.2701719, 0.2141893, -0.2037310, 0.1504426, -0.4206146, 0.4179203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 214

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B1_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7932799, upper bound: 0.7784622
time: 1.77 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7892751, upper bound: 0.7783456
time: 1.54 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1567600, 0.1568212, -0.1309368, 0.1281412, -0.2849012, 0.2877580
1: -0.1064691, 0.1189440, -0.0844008, 0.1036976, -0.2101667, 0.2033448
2: -0.1024963, 0.1829841, -0.0869065, 0.1536745, -0.2561708, 0.2698906
3: 0.3797373, 1.0415432, 0.4307910, 1.0375489, -0.6578116, 0.6107522
4: -0.1248855, 0.1212094, -0.0965243, 0.1019665, -0.2268521, 0.2177337
5: -0.0589422, 0.4962245, -0.0450168, 0.4547700, -0.5137122, 0.5412412
6: -0.1345023, 0.1552241, -0.1126368, 0.1283315, -0.2628338, 0.2678608
7: -0.2018844, 0.1412248, -0.1807818, 0.1181066, -0.3199910, 0.3220066
8: -0.1062965, 0.1790221, -0.0848394, 0.1516475, -0.2579440, 0.2638615
9: -0.2366939, 0.1756240, -0.2140292, 0.1530861, -0.3897800, 0.3896531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
time: 1.43 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
time: 1.43 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1567600, 0.1568212, -0.1097713, 0.1094665, -0.2662265, 0.2665925
1: -0.1064691, 0.1189440, -0.0735501, 0.0906561, -0.1971252, 0.1924941
2: -0.1024963, 0.1829841, -0.0781366, 0.1376066, -0.2401029, 0.2611207
3: 0.3797373, 1.0415432, 0.4806692, 1.0328857, -0.6531484, 0.5608740
4: -0.1248855, 0.1212094, -0.0836147, 0.0900388, -0.2149243, 0.2048241
5: -0.0589422, 0.4962245, -0.0389465, 0.4114023, -0.4703445, 0.5351709
6: -0.1345023, 0.1552241, -0.0977515, 0.1085226, -0.2430249, 0.2529756
7: -0.2018844, 0.1412248, -0.1598135, 0.1070034, -0.3088879, 0.3010383
8: -0.1062965, 0.1790221, -0.0671028, 0.1307158, -0.2370123, 0.2461250
9: -0.2366939, 0.1756240, -0.1890507, 0.1444671, -0.3811610, 0.3646747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7915055
time: 1.38 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7915430
time: 1.72 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1808042, 0.1795978, -0.1309908, 0.1281774, -0.3089815, 0.3105886
1: -0.1240668, 0.1353924, -0.0844159, 0.1037131, -0.2277799, 0.2198083
2: -0.1151550, 0.2080397, -0.0869208, 0.1537104, -0.2688654, 0.2949604
3: 0.3288566, 1.0454627, 0.4307021, 1.0375774, -0.7087208, 0.6147606
4: -0.1472030, 0.1383668, -0.0965327, 0.1019828, -0.2491858, 0.2348995
5: -0.0753658, 0.5354205, -0.0450237, 0.4548396, -0.5302054, 0.5804442
6: -0.1529576, 0.1779296, -0.1126591, 0.1283730, -0.2813306, 0.2905887
7: -0.2247660, 0.1558320, -0.1808241, 0.1181284, -0.3428944, 0.3366562
8: -0.1285102, 0.2051155, -0.0848854, 0.1516998, -0.2802100, 0.2900009
9: -0.2597371, 0.2049792, -0.2140917, 0.1531248, -0.4128619, 0.4190710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 137

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7914549
time: 1.40 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7915029
time: 1.55 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1808042, 0.1795978, -0.1098159, 0.1094803, -0.2902845, 0.2894136
1: -0.1240668, 0.1353924, -0.0735609, 0.0906679, -0.2147348, 0.2089533
2: -0.1151550, 0.2080397, -0.0781461, 0.1376354, -0.2527904, 0.2861858
3: 0.3288566, 1.0454627, 0.4806039, 1.0329114, -0.7040548, 0.5648588
4: -0.1472030, 0.1383668, -0.0836204, 0.0900511, -0.2372542, 0.2219872
5: -0.0753658, 0.5354205, -0.0389525, 0.4114621, -0.4868278, 0.5743730
6: -0.1529576, 0.1779296, -0.0977685, 0.1085547, -0.2615122, 0.2756981
7: -0.2247660, 0.1558320, -0.1598488, 0.1070204, -0.3317865, 0.3156808
8: -0.1285102, 0.2051155, -0.0671439, 0.1307577, -0.2592679, 0.2722594
9: -0.2597371, 0.2049792, -0.1891047, 0.1444957, -0.4042329, 0.3940839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7914549
time: 1.37 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7915029
time: 1.37 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1871822, 0.1848222, -0.1444001, 0.1423945, -0.3295766, 0.3292224
1: -0.1268898, 0.1400138, -0.0942667, 0.1123248, -0.2392146, 0.2342805
2: -0.1172737, 0.2127987, -0.0943482, 0.1676528, -0.2849265, 0.3071469
3: 0.3121817, 1.0468720, 0.4033424, 1.0380744, -0.7258927, 0.6435297
4: -0.1507959, 0.1417726, -0.1089849, 0.1113625, -0.2621584, 0.2507574
5: -0.0780900, 0.5500838, -0.0503971, 0.4790069, -0.5570969, 0.6004809
6: -0.1570138, 0.1832445, -0.1239658, 0.1420332, -0.2990470, 0.3072104
7: -0.2322025, 0.1572856, -0.1918564, 0.1288532, -0.3610558, 0.3491420
8: -0.1351245, 0.2106682, -0.0945732, 0.1655031, -0.3006276, 0.3052414
9: -0.2676096, 0.2100053, -0.2268856, 0.1594968, -0.4271064, 0.4368909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034553
time: 1.40 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034553
time: 1.53 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1872540, 0.1848790, -0.1696174, 0.1665345, -0.3537885, 0.3544964
1: -0.1269217, 0.1400379, -0.1090926, 0.1265446, -0.2534663, 0.2491304
2: -0.1173014, 0.2128521, -0.1048818, 0.1919379, -0.3092392, 0.3177340
3: 0.3120536, 1.0469080, 0.3475761, 1.0425621, -0.7305085, 0.6993319
4: -0.1508223, 0.1418017, -0.1278438, 0.1253087, -0.2761310, 0.2696455
5: -0.0781112, 0.5501719, -0.0585232, 0.5250446, -0.6031558, 0.6086951
6: -0.1570494, 0.1833043, -0.1411720, 0.1663494, -0.3233989, 0.3244763
7: -0.2322623, 0.1573222, -0.2154799, 0.1434929, -0.3757552, 0.3728021
8: -0.1351912, 0.2107452, -0.1177682, 0.1899399, -0.3251311, 0.3285134
9: -0.2676838, 0.2100775, -0.2538672, 0.1776035, -0.4452872, 0.4639447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8032636
time: 1.61 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8034263
time: 1.39 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1871822, 0.1848222, -0.1233806, 0.1248877, -0.3120698, 0.3082028
1: -0.1268898, 0.1400138, -0.0834162, 0.0998542, -0.2267440, 0.2234300
2: -0.1172737, 0.2127987, -0.0860733, 0.1495237, -0.2667974, 0.2988719
3: 0.3121817, 1.0468720, 0.4511277, 1.0340265, -0.7218448, 0.5957443
4: -0.1507959, 0.1417726, -0.0964279, 0.0997690, -0.2505649, 0.2382005
5: -0.0780900, 0.5500838, -0.0446325, 0.4370301, -0.5151200, 0.5947163
6: -0.1570138, 0.1832445, -0.1097326, 0.1231070, -0.2801208, 0.2929771
7: -0.2322025, 0.1572856, -0.1720583, 0.1184527, -0.3506553, 0.3293439
8: -0.1351245, 0.2106682, -0.0767781, 0.1461016, -0.2812261, 0.2874462
9: -0.2676096, 0.2100053, -0.2033731, 0.1516391, -0.4192487, 0.4133785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7942464, upper bound: 0.7846487
time: 1.66 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7899403, upper bound: 0.7845176
time: 1.34 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1872540, 0.1848790, -0.1441203, 0.1428802, -0.3301342, 0.3289993
1: -0.1269217, 0.1400379, -0.0956449, 0.1114902, -0.2384119, 0.2356828
2: -0.1173014, 0.2128521, -0.0953584, 0.1692048, -0.2865061, 0.3082106
3: 0.3120536, 1.0469080, 0.4086236, 1.0377977, -0.7257441, 0.6382844
4: -0.1508223, 0.1418017, -0.1108642, 0.1118622, -0.2626845, 0.2526659
5: -0.0781112, 0.5501719, -0.0512303, 0.4733024, -0.5514136, 0.6014022
6: -0.1570494, 0.1833043, -0.1244279, 0.1425186, -0.2995681, 0.3077322
7: -0.2322623, 0.1573222, -0.1896065, 0.1316204, -0.3638827, 0.3469288
8: -0.1351912, 0.2107452, -0.0931671, 0.1664481, -0.3016392, 0.3039123
9: -0.2676838, 0.2100775, -0.2248872, 0.1614067, -0.4290904, 0.4349647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7941413, upper bound: 0.7826763
time: 1.87 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7898792, upper bound: 0.7825541
time: 1.45 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1541927, 0.1542200, -0.1533972, 0.1510524, -0.3052451, 0.3076172
1: -0.1043523, 0.1175890, -0.0999786, 0.1170783, -0.2214306, 0.2175676
2: -0.1012686, 0.1801942, -0.0986784, 0.1766899, -0.2779585, 0.2788726
3: 0.3851002, 1.0409769, 0.3860659, 1.0400602, -0.6549599, 0.6549109
4: -0.1223764, 0.1192424, -0.1159150, 0.1165973, -0.2389738, 0.2351573
5: -0.0572329, 0.4920993, -0.0534560, 0.4931713, -0.5504042, 0.5455553
6: -0.1325160, 0.1527448, -0.1303429, 0.1506161, -0.2831321, 0.2830877
7: -0.1995640, 0.1394854, -0.1990890, 0.1352538, -0.3348178, 0.3385744
8: -0.1037776, 0.1764105, -0.1019083, 0.1744351, -0.2782127, 0.2783188
9: -0.2343406, 0.1723778, -0.2356595, 0.1655525, -0.3998930, 0.4080374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7952745
time: 1.48 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7954979
time: 1.41 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1541927, 0.1542200, -0.1321557, 0.1327813, -0.2869740, 0.2863757
1: -0.1043523, 0.1175890, -0.0888848, 0.1045387, -0.2088910, 0.2064738
2: -0.1012686, 0.1801942, -0.0903805, 0.1579078, -0.2591764, 0.2705747
3: 0.3851002, 1.0409769, 0.4341635, 1.0359497, -0.6508495, 0.6068134
4: -0.1223764, 0.1192424, -0.1029130, 0.1050280, -0.2274045, 0.2221554
5: -0.0572329, 0.4920993, -0.0475690, 0.4510371, -0.5082700, 0.5396683
6: -0.1325160, 0.1527448, -0.1161193, 0.1313998, -0.2639159, 0.2688641
7: -0.1995640, 0.1394854, -0.1789801, 0.1248534, -0.3244174, 0.3184655
8: -0.1037776, 0.1764105, -0.0834902, 0.1549705, -0.2587481, 0.2599008
9: -0.2343406, 0.1723778, -0.2120077, 0.1565019, -0.3908425, 0.3843855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7952745
time: 1.65 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7954979
time: 1.44 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1227716, 0.1210608, -0.1943357, 0.1888304, -0.3116020, 0.3153964
1: -0.0809997, 0.0987565, -0.1257584, 0.1424667, -0.2234663, 0.2245149
2: -0.0840840, 0.1479988, -0.1160953, 0.2155464, -0.2996304, 0.2640941
3: 0.4501366, 1.0351803, 0.2920325, 1.0474887, -0.5973520, 0.7431479
4: -0.0927137, 0.0977414, -0.1470899, 0.1426782, -0.2353919, 0.2448313
5: -0.0431789, 0.4368751, -0.0747956, 0.5678933, -0.6110722, 0.5116707
6: -0.1074239, 0.1213654, -0.1592225, 0.1884658, -0.2958898, 0.2805879
7: -0.1722133, 0.1151237, -0.2394434, 0.1560574, -0.3282706, 0.3545672
8: -0.0774283, 0.1443600, -0.1405176, 0.2157013, -0.2931296, 0.2848776
9: -0.2038944, 0.1504426, -0.2786052, 0.2071444, -0.4110388, 0.4290479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7952164
time: 1.45 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7952164
time: 1.89 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.1441203, 0.1428802, -0.1943357, 0.1888304, -0.3329507, 0.3372159
1: -0.0956449, 0.1114902, -0.1257584, 0.1424667, -0.2381116, 0.2372486
2: -0.0953584, 0.1692048, -0.1160953, 0.2155464, -0.3109048, 0.2853000
3: 0.4086236, 1.0377977, 0.2920325, 1.0474887, -0.6388651, 0.7457652
4: -0.1108642, 0.1118622, -0.1470899, 0.1426782, -0.2535424, 0.2589521
5: -0.0512303, 0.4733024, -0.0747956, 0.5678933, -0.6191236, 0.5480980
6: -0.1244279, 0.1425186, -0.1592225, 0.1884658, -0.3128937, 0.3017411
7: -0.1896065, 0.1316204, -0.2394434, 0.1560574, -0.3456639, 0.3710639
8: -0.0931671, 0.1664481, -0.1405176, 0.2157013, -0.3088685, 0.3069656
9: -0.2248872, 0.1614067, -0.2786052, 0.2071444, -0.4320316, 0.4400119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_B1_A1_B2_A2_A2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7954565
time: 1.39 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7954565
time: 1.66 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1891992, 0.1882601, -0.1309368, 0.1281412, -0.3173404, 0.3191969
1: -0.1322145, 0.1438359, -0.0844008, 0.1036976, -0.2359121, 0.2282367
2: -0.1212468, 0.2184048, -0.0869065, 0.1536745, -0.2749212, 0.3053113
3: 0.3106217, 1.0453222, 0.4307910, 1.0375489, -0.7269272, 0.6145312
4: -0.1580321, 0.1466553, -0.0965243, 0.1019665, -0.2599986, 0.2431796
5: -0.0844376, 0.5493211, -0.0450168, 0.4547700, -0.5392076, 0.5943379
6: -0.1609185, 0.1861594, -0.1126368, 0.1283315, -0.2892500, 0.2987962
7: -0.2332463, 0.1635423, -0.1807818, 0.1181066, -0.3513529, 0.3443241
8: -0.1359954, 0.2146020, -0.0848394, 0.1516475, -0.2876429, 0.2994414
9: -0.2669669, 0.2201408, -0.2140292, 0.1530861, -0.4200530, 0.4341700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_B1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
time: 1.26 seconds

## Relational analysis of IS_B1_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_B1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
time: 1.49 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2407487, 0.2352244, -0.1309908, 0.1281774, -0.3689260, 0.3662152
1: -0.1693661, 0.1837009, -0.0844159, 0.1037131, -0.2730792, 0.2681168
2: -0.1500545, 0.2681361, -0.0869208, 0.1537104, -0.3037649, 0.3550569
3: 0.1919659, 1.0544883, 0.4307021, 1.0375774, -0.8456115, 0.6237862
4: -0.2028808, 0.1828448, -0.0965327, 0.1019828, -0.3048636, 0.2793774
5: -0.1226393, 0.6400744, -0.0450237, 0.4548396, -0.5774789, 0.6850981
6: -0.1986031, 0.2339395, -0.1126591, 0.1283730, -0.3269761, 0.3465986
7: -0.2856337, 0.2025118, -0.1808241, 0.1181284, -0.4037621, 0.3833359
8: -0.1844749, 0.2705387, -0.0848854, 0.1516998, -0.3361747, 0.3554241
9: -0.3205736, 0.2813492, -0.2140917, 0.1531248, -0.4736983, 0.4954409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
time: 3.17 seconds

## Relational analysis of IS_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
time: 1.29 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1563327, 0.1617481, -0.1309368, 0.1281412, -0.2844739, 0.2926848
1: -0.1114379, 0.1191366, -0.0844008, 0.1036976, -0.2151356, 0.2035373
2: -0.1051293, 0.1882259, -0.0869065, 0.1536745, -0.2588038, 0.2751324
3: 0.3894673, 1.0387748, 0.4307910, 1.0375489, -0.6480816, 0.6079838
4: -0.1321565, 0.1253476, -0.0965243, 0.1019665, -0.2341230, 0.2218718
5: -0.0639865, 0.4863889, -0.0450168, 0.4547700, -0.5187565, 0.5314056
6: -0.1373913, 0.1562762, -0.1126368, 0.1283315, -0.2657229, 0.2689130
7: -0.1984849, 0.1463451, -0.1807818, 0.1181066, -0.3165915, 0.3271269
8: -0.1048121, 0.1800739, -0.0848394, 0.1516475, -0.2564596, 0.2649133
9: -0.2310548, 0.1846232, -0.2140292, 0.1530861, -0.3841409, 0.3986523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_B1_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7844454, upper bound: 0.7914482
time: 1.42 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7842582, upper bound: 0.7855877
time: 1.82 seconds

## BFS IS instance: IS_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2035880, 0.2027279, -0.1309908, 0.1281774, -0.3317654, 0.3337187
1: -0.1435095, 0.1551237, -0.0844159, 0.1037131, -0.2472226, 0.2395396
2: -0.1306034, 0.2348222, -0.0869208, 0.1537104, -0.2843138, 0.3217430
3: 0.2792566, 1.0475086, 0.4307021, 1.0375774, -0.7583208, 0.6168065
4: -0.1733959, 0.1573512, -0.0965327, 0.1019828, -0.2753787, 0.2538839
5: -0.0953804, 0.5691186, -0.0450237, 0.4548396, -0.5502200, 0.6141422
6: -0.1728129, 0.2003046, -0.1126591, 0.1283730, -0.3011859, 0.3129637
7: -0.2465810, 0.1726674, -0.1808241, 0.1181284, -0.3647094, 0.3534915
8: -0.1487181, 0.2324937, -0.0848854, 0.1516998, -0.3004179, 0.3173790
9: -0.2787447, 0.2423473, -0.2140917, 0.1531248, -0.4318694, 0.4564391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_B1_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7829086, upper bound: 0.7914482
time: 1.81 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7827073, upper bound: 0.7855847
time: 1.50 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1860761, 0.1851600, -0.1533972, 0.1510524, -0.3371285, 0.3385572
1: -0.1293941, 0.1412639, -0.0999786, 0.1170783, -0.2464723, 0.2412425
2: -0.1190411, 0.2150141, -0.0986784, 0.1766899, -0.2957310, 0.3136925
3: 0.3175419, 1.0446627, 0.3860659, 1.0400602, -0.7225183, 0.6585968
4: -0.1545252, 0.1440135, -0.1159150, 0.1165973, -0.2711225, 0.2599284
5: -0.0813120, 0.5443576, -0.0534560, 0.4931713, -0.5744833, 0.5978136
6: -0.1583497, 0.1831143, -0.1303429, 0.1506161, -0.3089659, 0.3134573
7: -0.2300747, 0.1605430, -0.1990890, 0.1352538, -0.3653286, 0.3596320
8: -0.1329630, 0.2109061, -0.1019083, 0.1744351, -0.3073982, 0.3128144
9: -0.2639462, 0.2152676, -0.2356595, 0.1655525, -0.4294987, 0.4509271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
time: 1.78 seconds

## Relational analysis of IS_B1_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_B1_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
time: 1.49 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2372470, 0.2318721, -0.1534283, 0.1510626, -0.3883096, 0.3853003
1: -0.1663123, 0.1807386, -0.0999719, 0.1170793, -0.2833916, 0.2807105
2: -0.1476544, 0.2644214, -0.0986774, 0.1766946, -0.3243490, 0.3630989
3: 0.1993252, 1.0537640, 0.3860149, 1.0400901, -0.8407649, 0.6677491
4: -0.1989604, 0.1800209, -0.1158939, 0.1165963, -0.3155567, 0.2959148
5: -0.1192682, 0.6345442, -0.0534508, 0.4932092, -0.6124774, 0.6879950
6: -0.1958195, 0.2306706, -0.1303449, 0.1506327, -0.3464522, 0.3610155
7: -0.2821558, 0.1992482, -0.1991151, 0.1352518, -0.4174076, 0.3983633
8: -0.1811105, 0.2666074, -0.1019381, 0.1744641, -0.3555745, 0.3685455
9: -0.3173229, 0.2761179, -0.2357045, 0.1655724, -0.4828952, 0.5118225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=39, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
time: 1.49 seconds

## Relational analysis of IS_B1_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_B1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
time: 1.57 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1533816, 0.1588915, -0.1533972, 0.1510524, -0.3044340, 0.3122887
1: -0.1090367, 0.1168976, -0.0999786, 0.1170783, -0.2261149, 0.2168761
2: -0.1033165, 0.1849920, -0.0986784, 0.1766899, -0.2800065, 0.2836704
3: 0.3958269, 1.0381076, 0.3860659, 1.0400602, -0.6442332, 0.6520417
4: -0.1289763, 0.1230586, -0.1159150, 0.1165973, -0.2455736, 0.2389736
5: -0.0615832, 0.4817533, -0.0534560, 0.4931713, -0.5547544, 0.5352094
6: -0.1349590, 0.1534606, -0.1303429, 0.1506161, -0.2855752, 0.2838035
7: -0.1955649, 0.1443795, -0.1990890, 0.1352538, -0.3308187, 0.3434685
8: -0.1019360, 0.1767158, -0.1019083, 0.1744351, -0.2763711, 0.2786241
9: -0.2283872, 0.1802541, -0.2356595, 0.1655525, -0.3939397, 0.4159136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_B1_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7846493, upper bound: 0.7949075
time: 1.70 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_B1_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7845176, upper bound: 0.7901685
time: 1.35 seconds

## BFS IS instance: IS_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.2004521, 0.1995821, -0.1534283, 0.1510626, -0.3515148, 0.3530104
1: -0.1409338, 0.1524869, -0.0999719, 0.1170793, -0.2580131, 0.2524588
2: -0.1285217, 0.2313916, -0.0986774, 0.1766946, -0.3052163, 0.3300690
3: 0.2862448, 1.0468485, 0.3860149, 1.0400901, -0.7538453, 0.6608337
4: -0.1698820, 0.1549013, -0.1158939, 0.1165963, -0.2864783, 0.2707952
5: -0.0926658, 0.5640382, -0.0534508, 0.4932092, -0.5858750, 0.6174889
6: -0.1702059, 0.1972777, -0.1303449, 0.1506327, -0.3208385, 0.3276227
7: -0.2433751, 0.1705400, -0.1991151, 0.1352518, -0.3786269, 0.3696550
8: -0.1456267, 0.2287995, -0.1019381, 0.1744641, -0.3200908, 0.3307375
9: -0.2758648, 0.2374908, -0.2357045, 0.1655724, -0.4414371, 0.4731954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_B1_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7830842, upper bound: 0.7949075
time: 1.52 seconds

## Relational analysis of IS_B1_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7829569, upper bound: 0.7901672
time: 1.52 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1998923, 0.1982723, -0.1023928, 0.1014286, -0.3013209, 0.3006651
1: -0.1406930, 0.1518827, -0.0683402, 0.0861951, -0.2268881, 0.2202229
2: -0.1279270, 0.2296020, -0.0738268, 0.1308653, -0.2587923, 0.3034288
3: 0.2891118, 1.0477684, 0.4973913, 1.0313350, -0.7422231, 0.5503771
4: -0.1682882, 0.1547795, -0.0771136, 0.0863025, -0.2545907, 0.2318931
5: -0.0933137, 0.5648595, -0.0360820, 0.3978105, -0.4911243, 0.6009415
6: -0.1690394, 0.1961634, -0.0920873, 0.1003963, -0.2694357, 0.2882507
7: -0.2434054, 0.1728214, -0.1533401, 0.1005147, -0.3439201, 0.3261615
8: -0.1458922, 0.2268944, -0.0615834, 0.1234170, -0.2693092, 0.2884778
9: -0.2771135, 0.2350459, -0.1819123, 0.1395789, -0.4166925, 0.4169582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
time: 2.08 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
time: 1.99 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2000316, 0.1983934, -0.1227538, 0.1210608, -0.3210924, 0.3211472
1: -0.1407759, 0.1519596, -0.0809997, 0.0986160, -0.2393919, 0.2329593
2: -0.1279949, 0.2297264, -0.0840840, 0.1479863, -0.2759812, 0.3138104
3: 0.2888345, 1.0478166, 0.4501366, 1.0350879, -0.7462535, 0.5976800
4: -0.1683757, 0.1548591, -0.0927137, 0.0963777, -0.2647533, 0.2475728
5: -0.0933857, 0.5650566, -0.0431789, 0.4368120, -0.5301977, 0.6082355
6: -0.1691267, 0.1962866, -0.1068448, 0.1213654, -0.2904921, 0.3031315
7: -0.2435319, 0.1729126, -0.1719452, 0.1151237, -0.3586556, 0.3448578
8: -0.1460141, 0.2270492, -0.0769005, 0.1443600, -0.2903741, 0.3039497
9: -0.2772529, 0.2352136, -0.2037310, 0.1504426, -0.4276956, 0.4389446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
time: 1.71 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
time: 1.62 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1563327, 0.1617481, -0.1097713, 0.1094665, -0.2657992, 0.2715194
1: -0.1114379, 0.1191366, -0.0735501, 0.0906561, -0.2020941, 0.1926867
2: -0.1051293, 0.1882259, -0.0781366, 0.1376066, -0.2427359, 0.2663625
3: 0.3894673, 1.0387748, 0.4806692, 1.0328857, -0.6434184, 0.5581056
4: -0.1321565, 0.1253476, -0.0836147, 0.0900388, -0.2221952, 0.2089623
5: -0.0639865, 0.4863889, -0.0389465, 0.4114023, -0.4753888, 0.5253353
6: -0.1373913, 0.1562762, -0.0977515, 0.1085226, -0.2459140, 0.2540278
7: -0.1984849, 0.1463451, -0.1598135, 0.1070034, -0.3054884, 0.3061586
8: -0.1048121, 0.1800739, -0.0671028, 0.1307158, -0.2355278, 0.2471767
9: -0.2310548, 0.1846232, -0.1890507, 0.1444671, -0.3755219, 0.3736739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
time: 1.35 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
time: 1.59 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2035880, 0.2027279, -0.1098159, 0.1094803, -0.3130683, 0.3125437
1: -0.1435095, 0.1551237, -0.0735609, 0.0906679, -0.2341775, 0.2286847
2: -0.1306034, 0.2348222, -0.0781461, 0.1376354, -0.2682388, 0.3129683
3: 0.2792566, 1.0475086, 0.4806039, 1.0329114, -0.7536548, 0.5669047
4: -0.1733959, 0.1573512, -0.0836204, 0.0900511, -0.2634470, 0.2409717
5: -0.0953804, 0.5691186, -0.0389525, 0.4114621, -0.5068425, 0.6080710
6: -0.1728129, 0.2003046, -0.0977685, 0.1085547, -0.2813676, 0.2980731
7: -0.2465810, 0.1726674, -0.1598488, 0.1070204, -0.3536015, 0.3325162
8: -0.1487181, 0.2324937, -0.0671439, 0.1307577, -0.2794758, 0.2996375
9: -0.2787447, 0.2423473, -0.1891047, 0.1444957, -0.4232404, 0.4314520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_B1_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7914692
time: 1.65 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7915088
time: 1.75 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1967053, 0.1951568, -0.1233806, 0.1248877, -0.3215930, 0.3185374
1: -0.1378591, 0.1491317, -0.0834162, 0.0998542, -0.2377133, 0.2325479
2: -0.1257100, 0.2261128, -0.0860733, 0.1495237, -0.2752337, 0.3121860
3: 0.2960573, 1.0470585, 0.4511277, 1.0340265, -0.7379692, 0.5959307
4: -0.1646512, 0.1521166, -0.0964279, 0.0997690, -0.2644202, 0.2485445
5: -0.0901681, 0.5598264, -0.0446325, 0.4370301, -0.5271982, 0.6044589
6: -0.1664458, 0.1931059, -0.1097326, 0.1231070, -0.2895529, 0.3028385
7: -0.2401468, 0.1698043, -0.1720583, 0.1184527, -0.3585995, 0.3418626
8: -0.1427681, 0.2231773, -0.0767781, 0.1461016, -0.2888697, 0.2999553
9: -0.2740694, 0.2301508, -0.2033731, 0.1516391, -0.4257086, 0.4335239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=35, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
time: 1.62 seconds

## Relational analysis of IS_B1_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
time: 2.26 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1968456, 0.1952789, -0.1441203, 0.1428802, -0.3397259, 0.3393992
1: -0.1379429, 0.1492092, -0.0956449, 0.1114902, -0.2494332, 0.2448541
2: -0.1257785, 0.2262378, -0.0953584, 0.1692048, -0.2949832, 0.3215962
3: 0.2957767, 1.0471070, 0.4086236, 1.0377977, -0.7420210, 0.6384834
4: -0.1647396, 0.1521965, -0.1108642, 0.1118622, -0.2766018, 0.2630607
5: -0.0902407, 0.5600255, -0.0512303, 0.4733024, -0.5635430, 0.6112558
6: -0.1665336, 0.1932303, -0.1244279, 0.1425186, -0.3090522, 0.3176581
7: -0.2402743, 0.1698961, -0.1896065, 0.1316204, -0.3718947, 0.3595026
8: -0.1428956, 0.2233331, -0.0931671, 0.1664481, -0.3093436, 0.3165002
9: -0.2742096, 0.2303201, -0.2248872, 0.1614067, -0.4356163, 0.4552072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B1_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
time: 1.85 seconds

## Relational analysis of IS_B1_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
time: 2.14 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1533816, 0.1588915, -0.1321557, 0.1327813, -0.2861629, 0.2910472
1: -0.1090367, 0.1168976, -0.0888848, 0.1045387, -0.2135754, 0.2057823
2: -0.1033165, 0.1849920, -0.0903805, 0.1579078, -0.2612243, 0.2753725
3: 0.3958269, 1.0381076, 0.4341635, 1.0359497, -0.6401228, 0.6039442
4: -0.1289763, 0.1230586, -0.1029130, 0.1050280, -0.2340044, 0.2259716
5: -0.0615832, 0.4817533, -0.0475690, 0.4510371, -0.5126203, 0.5293223
6: -0.1349590, 0.1534606, -0.1161193, 0.1313998, -0.2663589, 0.2695799
7: -0.1955649, 0.1443795, -0.1789801, 0.1248534, -0.3204183, 0.3233596
8: -0.1019360, 0.1767158, -0.0834902, 0.1549705, -0.2569064, 0.2602060
9: -0.2283872, 0.1802541, -0.2120077, 0.1565019, -0.3848891, 0.3922618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7966857, upper bound: 0.7954585
time: 1.54 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7966841, upper bound: 0.7944662
time: 2.99 seconds

## BFS IS instance: IS_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.2004521, 0.1995821, -0.1321754, 0.1327666, -0.3332188, 0.3317575
1: -0.1409338, 0.1524869, -0.0888752, 0.1045362, -0.2454700, 0.2413621
2: -0.1285217, 0.2313916, -0.0903756, 0.1579055, -0.2864272, 0.3217672
3: 0.2862448, 1.0468485, 0.4341351, 1.0359751, -0.7497303, 0.6127134
4: -0.1698820, 0.1549013, -0.1028938, 0.1050225, -0.2749045, 0.2577951
5: -0.0926658, 0.5640382, -0.0475642, 0.4510593, -0.5437251, 0.6116023
6: -0.1702059, 0.1972777, -0.1161157, 0.1314074, -0.3016133, 0.3133934
7: -0.2433751, 0.1705400, -0.1789997, 0.1248456, -0.3682207, 0.3495397
8: -0.1456267, 0.2287995, -0.0835146, 0.1549872, -0.3006139, 0.3123140
9: -0.2758648, 0.2374908, -0.2120414, 0.1565140, -0.4323787, 0.4495323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B1_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7944233, upper bound: 0.7954348
time: 1.32 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7944205, upper bound: 0.7944312
time: 1.66 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1309368, 0.1281412, -0.1891992, 0.1882601, -0.3191969, 0.3173404
1: -0.0844008, 0.1036976, -0.1322145, 0.1438359, -0.2282367, 0.2359121
2: -0.0869065, 0.1536745, -0.1212468, 0.2184048, -0.3053113, 0.2749212
3: 0.4307910, 1.0375489, 0.3106217, 1.0453222, -0.6145312, 0.7269272
4: -0.0965243, 0.1019665, -0.1580321, 0.1466553, -0.2431796, 0.2599986
5: -0.0450168, 0.4547700, -0.0844376, 0.5493211, -0.5943379, 0.5392076
6: -0.1126368, 0.1283315, -0.1609185, 0.1861594, -0.2987962, 0.2892500
7: -0.1807818, 0.1181066, -0.2332463, 0.1635423, -0.3443241, 0.3513529
8: -0.0848394, 0.1516475, -0.1359954, 0.2146020, -0.2994414, 0.2876429
9: -0.2140292, 0.1530861, -0.2669669, 0.2201408, -0.4341700, 0.4200530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
time: 1.34 seconds

## Relational analysis of IS_B2_A1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
time: 1.49 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1309908, 0.1281774, -0.2407487, 0.2352244, -0.3662152, 0.3689260
1: -0.0844159, 0.1037131, -0.1693661, 0.1837009, -0.2681168, 0.2730792
2: -0.0869208, 0.1537104, -0.1500545, 0.2681361, -0.3550569, 0.3037649
3: 0.4307021, 1.0375774, 0.1919659, 1.0544883, -0.6237862, 0.8456115
4: -0.0965327, 0.1019828, -0.2028808, 0.1828448, -0.2793774, 0.3048636
5: -0.0450237, 0.4548396, -0.1226393, 0.6400744, -0.6850981, 0.5774789
6: -0.1126591, 0.1283730, -0.1986031, 0.2339395, -0.3465986, 0.3269761
7: -0.1808241, 0.1181284, -0.2856337, 0.2025118, -0.3833359, 0.4037621
8: -0.0848854, 0.1516998, -0.1844749, 0.2705387, -0.3554241, 0.3361747
9: -0.2140917, 0.1531248, -0.3205736, 0.2813492, -0.4954409, 0.4736983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
time: 1.45 seconds

## Relational analysis of IS_B2_A1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
time: 1.42 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1309368, 0.1281412, -0.1563327, 0.1617481, -0.2926848, 0.2844739
1: -0.0844008, 0.1036976, -0.1114379, 0.1191366, -0.2035373, 0.2151356
2: -0.0869065, 0.1536745, -0.1051293, 0.1882259, -0.2751324, 0.2588038
3: 0.4307910, 1.0375489, 0.3894673, 1.0387748, -0.6079838, 0.6480816
4: -0.0965243, 0.1019665, -0.1321565, 0.1253476, -0.2218718, 0.2341230
5: -0.0450168, 0.4547700, -0.0639865, 0.4863889, -0.5314056, 0.5187565
6: -0.1126368, 0.1283315, -0.1373913, 0.1562762, -0.2689130, 0.2657229
7: -0.1807818, 0.1181066, -0.1984849, 0.1463451, -0.3271269, 0.3165915
8: -0.0848394, 0.1516475, -0.1048121, 0.1800739, -0.2649133, 0.2564596
9: -0.2140292, 0.1530861, -0.2310548, 0.1846232, -0.3986523, 0.3841409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B2_A1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914482, upper bound: 0.7844454
time: 1.55 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7855877, upper bound: 0.7842583
time: 1.60 seconds

## BFS IS instance: IS_B2_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1309908, 0.1281774, -0.2035880, 0.2027279, -0.3337187, 0.3317654
1: -0.0844159, 0.1037131, -0.1435095, 0.1551237, -0.2395396, 0.2472226
2: -0.0869208, 0.1537104, -0.1306034, 0.2348222, -0.3217430, 0.2843138
3: 0.4307021, 1.0375774, 0.2792566, 1.0475086, -0.6168065, 0.7583208
4: -0.0965327, 0.1019828, -0.1733959, 0.1573512, -0.2538839, 0.2753787
5: -0.0450237, 0.4548396, -0.0953804, 0.5691186, -0.6141422, 0.5502200
6: -0.1126591, 0.1283730, -0.1728129, 0.2003046, -0.3129637, 0.3011859
7: -0.1808241, 0.1181284, -0.2465810, 0.1726674, -0.3534915, 0.3647094
8: -0.0848854, 0.1516998, -0.1487181, 0.2324937, -0.3173790, 0.3004179
9: -0.2140917, 0.1531248, -0.2787447, 0.2423473, -0.4564391, 0.4318694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B2_A1_A1_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914482, upper bound: 0.7829086
time: 1.66 seconds

## Relational analysis of IS_B2_A1_A1_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7855847, upper bound: 0.7827073
time: 1.64 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1533972, 0.1510524, -0.1860761, 0.1851600, -0.3385572, 0.3371285
1: -0.0999786, 0.1170783, -0.1293941, 0.1412639, -0.2412425, 0.2464723
2: -0.0986784, 0.1766899, -0.1190411, 0.2150141, -0.3136925, 0.2957310
3: 0.3860659, 1.0400602, 0.3175419, 1.0446627, -0.6585968, 0.7225183
4: -0.1159150, 0.1165973, -0.1545252, 0.1440135, -0.2599284, 0.2711225
5: -0.0534560, 0.4931713, -0.0813120, 0.5443576, -0.5978136, 0.5744833
6: -0.1303429, 0.1506161, -0.1583497, 0.1831143, -0.3134573, 0.3089659
7: -0.1990890, 0.1352538, -0.2300747, 0.1605430, -0.3596320, 0.3653286
8: -0.1019083, 0.1744351, -0.1329630, 0.2109061, -0.3128144, 0.3073982
9: -0.2356595, 0.1655525, -0.2639462, 0.2152676, -0.4509271, 0.4294987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A2_B1_B1_A1

### Relational analysis result of IS_B2_A1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
time: 1.39 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_B1_A2

### Relational analysis result of IS_B2_A1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
time: 1.62 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1534283, 0.1510626, -0.2372470, 0.2318721, -0.3853003, 0.3883096
1: -0.0999719, 0.1170793, -0.1663123, 0.1807386, -0.2807105, 0.2833916
2: -0.0986774, 0.1766946, -0.1476544, 0.2644214, -0.3630989, 0.3243490
3: 0.3860149, 1.0400901, 0.1993252, 1.0537640, -0.6677491, 0.8407649
4: -0.1158939, 0.1165963, -0.1989604, 0.1800209, -0.2959148, 0.3155567
5: -0.0534508, 0.4932092, -0.1192682, 0.6345442, -0.6879950, 0.6124774
6: -0.1303449, 0.1506327, -0.1958195, 0.2306706, -0.3610155, 0.3464522
7: -0.1991151, 0.1352518, -0.2821558, 0.1992482, -0.3983633, 0.4174076
8: -0.1019381, 0.1744641, -0.1811105, 0.2666074, -0.3685455, 0.3555745
9: -0.2357045, 0.1655724, -0.3173229, 0.2761179, -0.5118225, 0.4828952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=39, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A1_A2_B1_B2_A1

### Relational analysis result of IS_B2_A1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
time: 1.50 seconds

## Relational analysis of IS_B2_A1_A1_A2_B1_B2_A2

### Relational analysis result of IS_B2_A1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
time: 1.54 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1533972, 0.1510524, -0.1533816, 0.1588915, -0.3122887, 0.3044340
1: -0.0999786, 0.1170783, -0.1090367, 0.1168976, -0.2168761, 0.2261149
2: -0.0986784, 0.1766899, -0.1033165, 0.1849920, -0.2836704, 0.2800065
3: 0.3860659, 1.0400602, 0.3958269, 1.0381076, -0.6520417, 0.6442332
4: -0.1159150, 0.1165973, -0.1289763, 0.1230586, -0.2389736, 0.2455736
5: -0.0534560, 0.4931713, -0.0615832, 0.4817533, -0.5352094, 0.5547544
6: -0.1303429, 0.1506161, -0.1349590, 0.1534606, -0.2838035, 0.2855752
7: -0.1990890, 0.1352538, -0.1955649, 0.1443795, -0.3434685, 0.3308187
8: -0.1019083, 0.1744351, -0.1019360, 0.1767158, -0.2786241, 0.2763711
9: -0.2356595, 0.1655525, -0.2283872, 0.1802541, -0.4159136, 0.3939397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 71

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B2_A1_A1_A2_B2_B1_A1

### Relational analysis result of IS_B2_A1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7949075, upper bound: 0.7846493
time: 1.46 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_B1_A2

### Relational analysis result of IS_B2_A1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7901685, upper bound: 0.7845176
time: 1.59 seconds

## BFS IS instance: IS_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1534283, 0.1510626, -0.2004521, 0.1995821, -0.3530104, 0.3515148
1: -0.0999719, 0.1170793, -0.1409338, 0.1524869, -0.2524588, 0.2580131
2: -0.0986774, 0.1766946, -0.1285217, 0.2313916, -0.3300690, 0.3052163
3: 0.3860149, 1.0400901, 0.2862448, 1.0468485, -0.6608337, 0.7538453
4: -0.1158939, 0.1165963, -0.1698820, 0.1549013, -0.2707952, 0.2864783
5: -0.0534508, 0.4932092, -0.0926658, 0.5640382, -0.6174889, 0.5858750
6: -0.1303449, 0.1506327, -0.1702059, 0.1972777, -0.3276227, 0.3208385
7: -0.1991151, 0.1352518, -0.2433751, 0.1705400, -0.3696550, 0.3786269
8: -0.1019381, 0.1744641, -0.1456267, 0.2287995, -0.3307375, 0.3200908
9: -0.2357045, 0.1655724, -0.2758648, 0.2374908, -0.4731954, 0.4414371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B2_A1_A1_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7949075, upper bound: 0.7830842
time: 1.98 seconds

## Relational analysis of IS_B2_A1_A1_A2_B2_B2_A2

### Relational analysis result of IS_B2_A1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7901672, upper bound: 0.7829569
time: 1.70 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1023928, 0.1014286, -0.1998923, 0.1982723, -0.3006651, 0.3013209
1: -0.0683402, 0.0861951, -0.1406930, 0.1518827, -0.2202229, 0.2268881
2: -0.0738268, 0.1308653, -0.1279270, 0.2296020, -0.3034288, 0.2587923
3: 0.4973913, 1.0313350, 0.2891118, 1.0477684, -0.5503771, 0.7422231
4: -0.0771136, 0.0863025, -0.1682882, 0.1547795, -0.2318931, 0.2545907
5: -0.0360820, 0.3978105, -0.0933137, 0.5648595, -0.6009415, 0.4911243
6: -0.0920873, 0.1003963, -0.1690394, 0.1961634, -0.2882507, 0.2694357
7: -0.1533401, 0.1005147, -0.2434054, 0.1728214, -0.3261615, 0.3439201
8: -0.0615834, 0.1234170, -0.1458922, 0.2268944, -0.2884778, 0.2693092
9: -0.1819123, 0.1395789, -0.2771135, 0.2350459, -0.4169582, 0.4166925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
time: 1.38 seconds

## Relational analysis of IS_B2_A1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
time: 1.50 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1227538, 0.1210608, -0.2000316, 0.1983934, -0.3211472, 0.3210924
1: -0.0809997, 0.0986160, -0.1407759, 0.1519596, -0.2329593, 0.2393919
2: -0.0840840, 0.1479863, -0.1279949, 0.2297264, -0.3138104, 0.2759812
3: 0.4501366, 1.0350879, 0.2888345, 1.0478166, -0.5976800, 0.7462535
4: -0.0927137, 0.0963777, -0.1683757, 0.1548591, -0.2475728, 0.2647533
5: -0.0431789, 0.4368120, -0.0933857, 0.5650566, -0.6082355, 0.5301977
6: -0.1068448, 0.1213654, -0.1691267, 0.1962866, -0.3031315, 0.2904921
7: -0.1719452, 0.1151237, -0.2435319, 0.1729126, -0.3448578, 0.3586556
8: -0.0769005, 0.1443600, -0.1460141, 0.2270492, -0.3039497, 0.2903741
9: -0.2037310, 0.1504426, -0.2772529, 0.2352136, -0.4389446, 0.4276956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
time: 1.46 seconds

## Relational analysis of IS_B2_A1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
time: 1.34 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1097713, 0.1094665, -0.1563327, 0.1617481, -0.2715194, 0.2657992
1: -0.0735501, 0.0906561, -0.1114379, 0.1191366, -0.1926867, 0.2020941
2: -0.0781366, 0.1376066, -0.1051293, 0.1882259, -0.2663625, 0.2427359
3: 0.4806692, 1.0328857, 0.3894673, 1.0387748, -0.5581056, 0.6434184
4: -0.0836147, 0.0900388, -0.1321565, 0.1253476, -0.2089623, 0.2221952
5: -0.0389465, 0.4114023, -0.0639865, 0.4863889, -0.5253353, 0.4753888
6: -0.0977515, 0.1085226, -0.1373913, 0.1562762, -0.2540278, 0.2459140
7: -0.1598135, 0.1070034, -0.1984849, 0.1463451, -0.3061586, 0.3054884
8: -0.0671028, 0.1307158, -0.1048121, 0.1800739, -0.2471767, 0.2355278
9: -0.1890507, 0.1444671, -0.2310548, 0.1846232, -0.3736739, 0.3755219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
time: 1.28 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
time: 1.23 seconds

## BFS IS instance: IS_B2_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1098159, 0.1094803, -0.2035880, 0.2027279, -0.3125437, 0.3130683
1: -0.0735609, 0.0906679, -0.1435095, 0.1551237, -0.2286847, 0.2341775
2: -0.0781461, 0.1376354, -0.1306034, 0.2348222, -0.3129683, 0.2682388
3: 0.4806039, 1.0329114, 0.2792566, 1.0475086, -0.5669047, 0.7536548
4: -0.0836204, 0.0900511, -0.1733959, 0.1573512, -0.2409717, 0.2634470
5: -0.0389525, 0.4114621, -0.0953804, 0.5691186, -0.6080710, 0.5068425
6: -0.0977685, 0.1085547, -0.1728129, 0.2003046, -0.2980731, 0.2813676
7: -0.1598488, 0.1070204, -0.2465810, 0.1726674, -0.3325162, 0.3536015
8: -0.0671439, 0.1307577, -0.1487181, 0.2324937, -0.2996375, 0.2794758
9: -0.1891047, 0.1444957, -0.2787447, 0.2423473, -0.4314520, 0.4232404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914692, upper bound: 0.7914549
time: 2.11 seconds

## Relational analysis of IS_B2_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7914692, upper bound: 0.7952164
time: 1.51 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1233806, 0.1248877, -0.1967053, 0.1951568, -0.3185374, 0.3215930
1: -0.0834162, 0.0998542, -0.1378591, 0.1491317, -0.2325479, 0.2377133
2: -0.0860733, 0.1495237, -0.1257100, 0.2261128, -0.3121860, 0.2752337
3: 0.4511277, 1.0340265, 0.2960573, 1.0470585, -0.5959307, 0.7379692
4: -0.0964279, 0.0997690, -0.1646512, 0.1521166, -0.2485445, 0.2644202
5: -0.0446325, 0.4370301, -0.0901681, 0.5598264, -0.6044589, 0.5271982
6: -0.1097326, 0.1231070, -0.1664458, 0.1931059, -0.3028385, 0.2895529
7: -0.1720583, 0.1184527, -0.2401468, 0.1698043, -0.3418626, 0.3585995
8: -0.0767781, 0.1461016, -0.1427681, 0.2231773, -0.2999553, 0.2888697
9: -0.2033731, 0.1516391, -0.2740694, 0.2301508, -0.4335239, 0.4257086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
time: 1.82 seconds

## Relational analysis of IS_B2_A1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
time: 1.59 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1441203, 0.1428802, -0.1968456, 0.1952789, -0.3393992, 0.3397259
1: -0.0956449, 0.1114902, -0.1379429, 0.1492092, -0.2448541, 0.2494332
2: -0.0953584, 0.1692048, -0.1257785, 0.2262378, -0.3215962, 0.2949832
3: 0.4086236, 1.0377977, 0.2957767, 1.0471070, -0.6384834, 0.7420210
4: -0.1108642, 0.1118622, -0.1647396, 0.1521965, -0.2630607, 0.2766018
5: -0.0512303, 0.4733024, -0.0902407, 0.5600255, -0.6112558, 0.5635430
6: -0.1244279, 0.1425186, -0.1665336, 0.1932303, -0.3176581, 0.3090522
7: -0.1896065, 0.1316204, -0.2402743, 0.1698961, -0.3595026, 0.3718947
8: -0.0931671, 0.1664481, -0.1428956, 0.2233331, -0.3165002, 0.3093436
9: -0.2248872, 0.1614067, -0.2742096, 0.2303201, -0.4552072, 0.4356163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
time: 1.54 seconds

## Relational analysis of IS_B2_A1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
time: 1.44 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1321557, 0.1327813, -0.1533816, 0.1588915, -0.2910472, 0.2861629
1: -0.0888848, 0.1045387, -0.1090367, 0.1168976, -0.2057823, 0.2135754
2: -0.0903805, 0.1579078, -0.1033165, 0.1849920, -0.2753725, 0.2612243
3: 0.4341635, 1.0359497, 0.3958269, 1.0381076, -0.6039442, 0.6401228
4: -0.1029130, 0.1050280, -0.1289763, 0.1230586, -0.2259716, 0.2340044
5: -0.0475690, 0.4510371, -0.0615832, 0.4817533, -0.5293223, 0.5126203
6: -0.1161193, 0.1313998, -0.1349590, 0.1534606, -0.2695799, 0.2663589
7: -0.1789801, 0.1248534, -0.1955649, 0.1443795, -0.3233596, 0.3204183
8: -0.0834902, 0.1549705, -0.1019360, 0.1767158, -0.2602060, 0.2569064
9: -0.2120077, 0.1565019, -0.2283872, 0.1802541, -0.3922618, 0.3848891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A1_A2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7954585, upper bound: 0.7966857
time: 1.42 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7944662, upper bound: 0.7966841
time: 1.40 seconds

## BFS IS instance: IS_B2_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1321754, 0.1327666, -0.2004521, 0.1995821, -0.3317575, 0.3332188
1: -0.0888752, 0.1045362, -0.1409338, 0.1524869, -0.2413621, 0.2454700
2: -0.0903756, 0.1579055, -0.1285217, 0.2313916, -0.3217672, 0.2864272
3: 0.4341351, 1.0359751, 0.2862448, 1.0468485, -0.6127134, 0.7497303
4: -0.1028938, 0.1050225, -0.1698820, 0.1549013, -0.2577951, 0.2749045
5: -0.0475642, 0.4510593, -0.0926658, 0.5640382, -0.6116023, 0.5437251
6: -0.1161157, 0.1314074, -0.1702059, 0.1972777, -0.3133934, 0.3016133
7: -0.1789997, 0.1248456, -0.2433751, 0.1705400, -0.3495397, 0.3682207
8: -0.0835146, 0.1549872, -0.1456267, 0.2287995, -0.3123140, 0.3006139
9: -0.2120414, 0.1565140, -0.2758648, 0.2374908, -0.4495323, 0.4323787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_B2_A1_A2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7954348, upper bound: 0.7944233
time: 1.33 seconds

## Relational analysis of IS_B2_A1_A2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7944312, upper bound: 0.7944205
time: 1.40 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1318198, 0.1321054, -0.1998923, 0.1982723, -0.3300921, 0.3319978
1: -0.0882853, 0.1050535, -0.1406930, 0.1518827, -0.2401680, 0.2457465
2: -0.0898393, 0.1568546, -0.1279270, 0.2296020, -0.3194413, 0.2847817
3: 0.4300937, 1.0355365, 0.2891118, 1.0477684, -0.6176746, 0.7464247
4: -0.1017791, 0.1047615, -0.1682882, 0.1547795, -0.2565585, 0.2730497
5: -0.0470315, 0.4541584, -0.0933137, 0.5648595, -0.6118910, 0.5474722
6: -0.1157565, 0.1309870, -0.1690394, 0.1961634, -0.3119199, 0.3000264
7: -0.1798957, 0.1234360, -0.2434054, 0.1728214, -0.3527172, 0.3668414
8: -0.0836543, 0.1540108, -0.1458922, 0.2268944, -0.3105487, 0.2999030
9: -0.2124019, 0.1563095, -0.2771135, 0.2350459, -0.4474478, 0.4334231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8027315, upper bound: 0.8000650
time: 1.45 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8027315, upper bound: 0.8033365
time: 1.56 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1802195, 0.1758185, -0.2000316, 0.1983934, -0.3786129, 0.3758502
1: -0.1147226, 0.1328505, -0.1407759, 0.1519596, -0.2666822, 0.2736264
2: -0.1100007, 0.2006704, -0.1279949, 0.2297264, -0.3397271, 0.3286653
3: 0.3217089, 1.0442970, 0.2888345, 1.0478166, -0.7261077, 0.7554625
4: -0.1316050, 0.1314472, -0.1683757, 0.1548591, -0.2864640, 0.2998229
5: -0.0607156, 0.5450642, -0.0933857, 0.5650566, -0.6257722, 0.6384499
6: -0.1487259, 0.1755292, -0.1691267, 0.1962866, -0.3450125, 0.3446559
7: -0.2232874, 0.1510086, -0.2435319, 0.1729126, -0.3962000, 0.3945404
8: -0.1227947, 0.2008618, -0.1460141, 0.2270492, -0.3498439, 0.3468758
9: -0.2646438, 0.1791402, -0.2772529, 0.2352136, -0.4998574, 0.4563931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8000204
time: 1.37 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8032667
time: 2.45 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1037636, 0.1073458, -0.1998923, 0.1982723, -0.3020359, 0.3072380
1: -0.0722249, 0.0861795, -0.1406930, 0.1518827, -0.2241076, 0.2268725
2: -0.0771485, 0.1325132, -0.1279270, 0.2296020, -0.3067505, 0.2604403
3: 0.5039725, 1.0289392, 0.2891118, 1.0477684, -0.5437958, 0.7398274
4: -0.0831220, 0.0881765, -0.1682882, 0.1547795, -0.2379014, 0.2564647
5: -0.0384876, 0.3913965, -0.0933137, 0.5648595, -0.6033471, 0.4847103
6: -0.0955334, 0.1021226, -0.1690394, 0.1961634, -0.2916968, 0.2711620
7: -0.1494413, 0.1071181, -0.2434054, 0.1728214, -0.3222627, 0.3505235
8: -0.0577507, 0.1305709, -0.1458922, 0.2268944, -0.2846451, 0.2764631
9: -0.1809110, 0.1422760, -0.2771135, 0.2350459, -0.4159570, 0.4193895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7939819, upper bound: 0.7989310
time: 1.52 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7939819, upper bound: 0.8020389
time: 1.49 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.1393587, 0.1442275, -0.2000316, 0.1983934, -0.3377521, 0.3442591
1: -0.0957595, 0.1119640, -0.1407759, 0.1519596, -0.2477192, 0.2527398
2: -0.0959855, 0.1656448, -0.1279949, 0.2297264, -0.3257120, 0.2936397
3: 0.4041806, 1.0366504, 0.2888345, 1.0478166, -0.6436360, 0.7478160
4: -0.1112557, 0.1112926, -0.1683757, 0.1548591, -0.2661148, 0.2796682
5: -0.0509032, 0.4739519, -0.0933857, 0.5650566, -0.6159598, 0.5673376
6: -0.1240363, 0.1428738, -0.1691267, 0.1962866, -0.3203229, 0.3120005
7: -0.1894059, 0.1332067, -0.2435319, 0.1729126, -0.3623185, 0.3767385
8: -0.0911180, 0.1595736, -0.1460141, 0.2270492, -0.3181672, 0.3055877
9: -0.2187822, 0.1642583, -0.2772529, 0.2352136, -0.4539959, 0.4415112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917048, upper bound: 0.7989163
time: 1.45 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7917048, upper bound: 0.8020113
time: 1.68 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1409265, 0.1401411, -0.1563327, 0.1617481, -0.3026746, 0.2964738
1: -0.0937277, 0.1098315, -0.1114379, 0.1191366, -0.2128643, 0.2212695
2: -0.0940710, 0.1656276, -0.1051293, 0.1882259, -0.2822969, 0.2707569
3: 0.4125848, 1.0375537, 0.3894673, 1.0387748, -0.6261900, 0.6480864
4: -0.1079966, 0.1099341, -0.1321565, 0.1253476, -0.2333441, 0.2420905
5: -0.0498775, 0.4687327, -0.0639865, 0.4863889, -0.5362664, 0.5327191
6: -0.1221122, 0.1394163, -0.1373913, 0.1562762, -0.2783884, 0.2768077
7: -0.1871469, 0.1296584, -0.1984849, 0.1463451, -0.3334920, 0.3281434
8: -0.0906981, 0.1630404, -0.1048121, 0.1800739, -0.2707720, 0.2678525
9: -0.2214662, 0.1613781, -0.2310548, 0.1846232, -0.4060894, 0.3924328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7940198
time: 1.69 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7978495
time: 2.09 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1410468, 0.1402437, -0.2035880, 0.2027279, -0.3437747, 0.3438317
1: -0.0937821, 0.1098841, -0.1435095, 0.1551237, -0.2489059, 0.2533936
2: -0.0941162, 0.1657250, -0.1306034, 0.2348222, -0.3289384, 0.2963284
3: 0.4123487, 1.0375963, 0.2792566, 1.0475086, -0.6351599, 0.7583398
4: -0.1080463, 0.1099874, -0.1733959, 0.1573512, -0.2653975, 0.2833833
5: -0.0499036, 0.4689205, -0.0953804, 0.5691186, -0.6190221, 0.5643010
6: -0.1221818, 0.1395204, -0.1728129, 0.2003046, -0.3224865, 0.3123333
7: -0.1872466, 0.1297250, -0.2465810, 0.1726674, -0.3599140, 0.3763061
8: -0.0907959, 0.1631610, -0.1487181, 0.2324937, -0.3232896, 0.3118792
9: -0.2215982, 0.1614619, -0.2787447, 0.2423473, -0.4639456, 0.4402066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7987457, upper bound: 0.7917143
time: 1.60 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7987457, upper bound: 0.7955412
time: 2.25 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1037636, 0.1073458, -0.1668676, 0.1710681, -0.2748317, 0.2742134
1: -0.0722249, 0.0861795, -0.1191725, 0.1268859, -0.1991107, 0.2053520
2: -0.0771485, 0.1325132, -0.1113653, 0.1992984, -0.2764469, 0.2438785
3: 0.5039725, 1.0289392, 0.3677489, 1.0410765, -0.5371040, 0.6611904
4: -0.0831220, 0.0881765, -0.1422104, 0.1328411, -0.2159630, 0.2303870
5: -0.0384876, 0.3913965, -0.0717306, 0.5018583, -0.5403458, 0.4631271
6: -0.0955334, 0.1021226, -0.1454507, 0.1661083, -0.2616417, 0.2475732
7: -0.1494413, 0.1071181, -0.2085314, 0.1530936, -0.3025348, 0.3156495
8: -0.0577507, 0.1305709, -0.1146384, 0.1922527, -0.2500035, 0.2452093
9: -0.1809110, 0.1422760, -0.2406706, 0.1994226, -0.3803337, 0.3829466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7915056
time: 1.28 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7952745
time: 1.46 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.1393587, 0.1442275, -0.1669926, 0.1711570, -0.3105157, 0.3112201
1: -0.0957595, 0.1119640, -0.1192446, 0.1269561, -0.2227156, 0.2312086
2: -0.0959855, 0.1656448, -0.1114245, 0.1994087, -0.2953942, 0.2770692
3: 0.4041806, 1.0366504, 0.3674909, 1.0411189, -0.6369382, 0.6691595
4: -0.1112557, 0.1112926, -0.1422892, 0.1329089, -0.2441647, 0.2535818
5: -0.0509032, 0.4739519, -0.0717918, 0.5020437, -0.5529469, 0.5457437
6: -0.1240363, 0.1428738, -0.1455303, 0.1662184, -0.2902546, 0.2884041
7: -0.1894059, 0.1332067, -0.2086468, 0.1531646, -0.3425705, 0.3418534
8: -0.0911180, 0.1595736, -0.1147540, 0.1923905, -0.2835086, 0.2743276
9: -0.2187822, 0.1642583, -0.2407961, 0.1995743, -0.4183566, 0.4050545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7904105, upper bound: 0.7951131
time: 2.01 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7904104, upper bound: 0.7941267
time: 1.40 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1484293, 0.1479993, -0.1967053, 0.1951568, -0.3435861, 0.3447046
1: -0.0993937, 0.1146939, -0.1378591, 0.1491317, -0.2485253, 0.2525530
2: -0.0980707, 0.1743052, -0.1257100, 0.2261128, -0.3241835, 0.3000153
3: 0.3986785, 1.0371230, 0.2960573, 1.0470585, -0.6483799, 0.7410657
4: -0.1152051, 0.1152600, -0.1646512, 0.1521166, -0.2673216, 0.2799112
5: -0.0530041, 0.4816796, -0.0901681, 0.5598264, -0.6128305, 0.5718477
6: -0.1283971, 0.1469685, -0.1664458, 0.1931059, -0.3215030, 0.3134143
7: -0.1931455, 0.1355090, -0.2401468, 0.1698043, -0.3629498, 0.3756557
8: -0.0960366, 0.1705458, -0.1427681, 0.2231773, -0.3192138, 0.3133138
9: -0.2284085, 0.1645007, -0.2740694, 0.2301508, -0.4585592, 0.4385702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 230

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_B2_A2_A2_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
time: 1.44 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
time: 1.40 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1962982, 0.1918369, -0.1968456, 0.1952789, -0.3915771, 0.3886825
1: -0.1299907, 0.1404513, -0.1379429, 0.1492092, -0.2791998, 0.2783943
2: -0.1164719, 0.2203114, -0.1257785, 0.2262378, -0.3427097, 0.3460898
3: 0.2917782, 1.0457768, 0.2957767, 1.0471070, -0.7553288, 0.7500002
4: -0.1502240, 0.1461449, -0.1647396, 0.1521965, -0.3024205, 0.3108845
5: -0.0757346, 0.5668938, -0.0902407, 0.5600255, -0.6357601, 0.6571344
6: -0.1615001, 0.1913464, -0.1665336, 0.1932303, -0.3547303, 0.3578801
7: -0.2381653, 0.1606373, -0.2402743, 0.1698961, -0.4080614, 0.4009115
8: -0.1406808, 0.2171511, -0.1428956, 0.2233331, -0.3640139, 0.3600467
9: -0.2777742, 0.2104512, -0.2742096, 0.2303201, -0.5080943, 0.4846608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A2_B1_A1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
time: 1.94 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8034341
time: 1.50 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1196315, 0.1256558, -0.1967053, 0.1951568, -0.3147883, 0.3223611
1: -0.0840786, 0.0971014, -0.1378591, 0.1491317, -0.2332103, 0.2349606
2: -0.0865548, 0.1481145, -0.1257100, 0.2261128, -0.3126676, 0.2738245
3: 0.4679182, 1.0307829, 0.2960573, 1.0470585, -0.5791403, 0.7347257
4: -0.0981833, 0.0997309, -0.1646512, 0.1521166, -0.2502999, 0.2643820
5: -0.0450296, 0.4223335, -0.0901681, 0.5598264, -0.6048560, 0.5125017
6: -0.1095581, 0.1200375, -0.1664458, 0.1931059, -0.3026640, 0.2864833
7: -0.1645265, 0.1209474, -0.2401468, 0.1698043, -0.3343309, 0.3610941
8: -0.0705832, 0.1461338, -0.1427681, 0.2231773, -0.2937604, 0.2889019
9: -0.1966162, 0.1516937, -0.2740694, 0.2301508, -0.4267669, 0.4257631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A2_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978170, upper bound: 0.7990156
time: 1.60 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978170, upper bound: 0.8024581
time: 1.52 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.1557610, 0.1594056, -0.1968456, 0.1952789, -0.3510399, 0.3562513
1: -0.1068809, 0.1209051, -0.1379429, 0.1492092, -0.2560901, 0.2588481
2: -0.1035944, 0.1865916, -0.1257785, 0.2262378, -0.3298323, 0.3123701
3: 0.3745235, 1.0382823, 0.2957767, 1.0471070, -0.6725835, 0.7425056
4: -0.1241746, 0.1207286, -0.1647396, 0.1521965, -0.2763711, 0.2854682
5: -0.0564940, 0.4995061, -0.0902407, 0.5600255, -0.6165196, 0.5897468
6: -0.1354058, 0.1584895, -0.1665336, 0.1932303, -0.3286361, 0.3250231
7: -0.2019843, 0.1445208, -0.2402743, 0.1698961, -0.3718804, 0.3847950
8: -0.1044977, 0.1724839, -0.1428956, 0.2233331, -0.3278308, 0.3153795
9: -0.2325112, 0.1720608, -0.2742096, 0.2303201, -0.4628313, 0.4462704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A2_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955412, upper bound: 0.7990125
time: 1.61 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7955412, upper bound: 0.8024551
time: 2.47 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1580102, 0.1571559, -0.1533816, 0.1588915, -0.3169017, 0.3105375
1: -0.1060455, 0.1194952, -0.1090367, 0.1168976, -0.2229430, 0.2285319
2: -0.1022715, 0.1842785, -0.1033165, 0.1849920, -0.2872635, 0.2875950
3: 0.3799411, 1.0392765, 0.3958269, 1.0381076, -0.6581665, 0.6434495
4: -0.1233590, 0.1214772, -0.1289763, 0.1230586, -0.2464176, 0.2504535
5: -0.0576574, 0.4963924, -0.0615832, 0.4817533, -0.5394108, 0.5579755
6: -0.1351836, 0.1561640, -0.1349590, 0.1534606, -0.2886442, 0.2911230
7: -0.2015463, 0.1416417, -0.1955649, 0.1443795, -0.3459258, 0.3372065
8: -0.1052095, 0.1800551, -0.1019360, 0.1767158, -0.2819253, 0.2819911
9: -0.2375896, 0.1750607, -0.2283872, 0.1802541, -0.4178437, 0.4034479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=36, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A2_B2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7940310
time: 1.51 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7980528
time: 1.76 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1581139, 0.1572413, -0.2004521, 0.1995821, -0.3576959, 0.3576935
1: -0.1060926, 0.1195356, -0.1409338, 0.1524869, -0.2585794, 0.2604694
2: -0.1023044, 0.1843603, -0.1285217, 0.2313916, -0.3336959, 0.3128820
3: 0.3797305, 1.0393201, 0.2862448, 1.0468485, -0.6671181, 0.7530754
4: -0.1234011, 0.1215250, -0.1698820, 0.1549013, -0.2783024, 0.2914070
5: -0.0576893, 0.4965519, -0.0926658, 0.5640382, -0.6217274, 0.5892177
6: -0.1352398, 0.1562515, -0.1702059, 0.1972777, -0.3325175, 0.3264574
7: -0.2016391, 0.1416883, -0.2433751, 0.1705400, -0.3721791, 0.3850634
8: -0.1053079, 0.1801580, -0.1456267, 0.2287995, -0.3341073, 0.3257847
9: -0.2377044, 0.1751693, -0.2758648, 0.2374908, -0.4751952, 0.4510341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=38, inp2_unstable=38, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 71

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_B2_A2_A2_B2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
time: 3.02 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7956726
time: 1.58 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.21 seconds
IS_B1_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
IS_B1_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
IS_B1_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
IS_B1_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000102
IS_B1_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7939819
IS_B1_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7940029
IS_B1_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7932799, upper bound: 0.7784622
IS_B1_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7892751, upper bound: 0.7783456
IS_B1_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
IS_B1_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7974645, upper bound: 0.7915430
IS_B1_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7915055
IS_B1_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7915430
IS_B1_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7914549
IS_B1_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7915029
IS_B1_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7914549
IS_B1_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7915029
IS_B1_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034553
IS_B1_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034553
IS_B1_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8032636
IS_B1_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8034263
IS_B1_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7942464, upper bound: 0.7846487
IS_B1_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7899403, upper bound: 0.7845176
IS_B1_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7941413, upper bound: 0.7826763
IS_B1_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7898792, upper bound: 0.7825541
IS_B1_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7952745
IS_B1_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7954979
IS_B1_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7952745
IS_B1_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7954979
IS_B1_A1_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7952164
IS_B1_A1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7952164
IS_B1_A1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7954565
IS_B1_A1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7954565
IS_B1_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
IS_B1_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
IS_B1_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
IS_B1_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
IS_B1_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7844454, upper bound: 0.7914482
IS_B1_A2_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7842582, upper bound: 0.7855877
IS_B1_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7829086, upper bound: 0.7914482
IS_B1_A2_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7827073, upper bound: 0.7855847
IS_B1_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
IS_B1_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
IS_B1_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
IS_B1_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
IS_B1_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7846493, upper bound: 0.7949075
IS_B1_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7845176, upper bound: 0.7901685
IS_B1_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7830842, upper bound: 0.7949075
IS_B1_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7829569, upper bound: 0.7901672
IS_B1_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
IS_B1_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
IS_B1_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
IS_B1_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
IS_B1_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
IS_B1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7952164, upper bound: 0.7915088
IS_B1_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7914692
IS_B1_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914549, upper bound: 0.7915088
IS_B1_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
IS_B1_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
IS_B1_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
IS_B1_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8023090, upper bound: 0.7957173
IS_B1_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7966857, upper bound: 0.7954585
IS_B1_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7966841, upper bound: 0.7944662
IS_B1_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7944233, upper bound: 0.7954348
IS_B1_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7944205, upper bound: 0.7944312
IS_B2_A1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
IS_B2_A1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
IS_B2_A1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
IS_B2_A1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8000183, upper bound: 0.8032636
IS_B2_A1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914482, upper bound: 0.7844454
IS_B2_A1_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7855877, upper bound: 0.7842583
IS_B2_A1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914482, upper bound: 0.7829086
IS_B2_A1_A1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7855847, upper bound: 0.7827073
IS_B2_A1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
IS_B2_A1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
IS_B2_A1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
IS_B2_A1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034615, upper bound: 0.8034553
IS_B2_A1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7949075, upper bound: 0.7846493
IS_B2_A1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7901685, upper bound: 0.7845176
IS_B2_A1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7949075, upper bound: 0.7830842
IS_B2_A1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7901672, upper bound: 0.7829569
IS_B2_A1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
IS_B2_A1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
IS_B2_A1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
IS_B2_A1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7917207, upper bound: 0.8018036
IS_B2_A1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
IS_B2_A1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7915088, upper bound: 0.7952164
IS_B2_A1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914692, upper bound: 0.7914549
IS_B2_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7914692, upper bound: 0.7952164
IS_B2_A1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
IS_B2_A1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
IS_B2_A1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
IS_B2_A1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7957173, upper bound: 0.8023090
IS_B2_A1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7954585, upper bound: 0.7966857
IS_B2_A1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7944662, upper bound: 0.7966841
IS_B2_A1_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7954348, upper bound: 0.7944233
IS_B2_A1_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7944312, upper bound: 0.7944205
IS_B2_A2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8027315, upper bound: 0.8000650
IS_B2_A2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8027315, upper bound: 0.8033365
IS_B2_A2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8000204
IS_B2_A2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7999972, upper bound: 0.8032667
IS_B2_A2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7939819, upper bound: 0.7989310
IS_B2_A2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7939819, upper bound: 0.8020389
IS_B2_A2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7917048, upper bound: 0.7989163
IS_B2_A2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7917048, upper bound: 0.8020113
IS_B2_A2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7940198
IS_B2_A2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7988044, upper bound: 0.7978495
IS_B2_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7987457, upper bound: 0.7917143
IS_B2_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7987457, upper bound: 0.7955412
IS_B2_A2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7915056
IS_B2_A2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7937048, upper bound: 0.7952745
IS_B2_A2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7904105, upper bound: 0.7951131
IS_B2_A2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7904104, upper bound: 0.7941267
IS_B2_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
IS_B2_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8034553, upper bound: 0.8034615
IS_B2_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8000183
IS_B2_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8032636, upper bound: 0.8034341
IS_B2_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7978170, upper bound: 0.7990156
IS_B2_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7978170, upper bound: 0.8024581
IS_B2_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7955412, upper bound: 0.7990125
IS_B2_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.7955412, upper bound: 0.8024551
IS_B2_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7940310
IS_B2_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8019176, upper bound: 0.7980528
IS_B2_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7917207
IS_B2_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.21
Output dim: 3, lower bound: -0.8018036, upper bound: 0.7956726
IS_B2_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 3, lower bound: -0.7978070, upper bound: 0.7955306
IS_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 3, lower bound: -0.7954968, upper bound: 0.7955000

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.19 + 596.43 = 600.62 seconds
