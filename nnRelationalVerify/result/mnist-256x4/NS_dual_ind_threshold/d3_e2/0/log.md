## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.7875191039999999


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.25 + 2.77 = 5.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.32 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.32
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.32
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.2500582, 0.2443756, -0.2592995, 0.2564672, -0.5065254, 0.5036751
1: -0.1734074, 0.1877671, -0.1868174, 0.1974913, -0.3708987, 0.3745845
2: -0.1552624, 0.2720147, -0.1666596, 0.2860169, -0.4412793, 0.4386743
3: 0.1661023, 1.0579869, 0.1520118, 1.0587583, -0.8926560, 0.9059751
4: -0.2047127, 0.1891258, -0.2195624, 0.2034553, -0.4081680, 0.4086882
5: -0.1254750, 0.6633909, -0.1395424, 0.6713460, -0.7968211, 0.8029333
6: -0.2023995, 0.2440670, -0.2119004, 0.2561235, -0.4585229, 0.4559675
7: -0.2983487, 0.2024147, -0.3077715, 0.2163616, -0.5147104, 0.5101862
8: -0.1971189, 0.2782628, -0.2069341, 0.2914328, -0.4885517, 0.4851969
9: -0.3337545, 0.2865764, -0.3397698, 0.3094461, -0.6432005, 0.6263462

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.64 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.50 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.2581915, 0.2553524, -0.2635888, 0.2611926, -0.5193841, 0.5189412
1: -0.1858926, 0.1966751, -0.1908717, 0.2009175, -0.3868101, 0.3875468
2: -0.1658475, 0.2849444, -0.1701133, 0.2907368, -0.4565842, 0.4550577
3: 0.1543276, 1.0586346, 0.1429092, 1.0597293, -0.9054018, 0.9157254
4: -0.2184964, 0.2024755, -0.2238975, 0.2077588, -0.4262551, 0.4263731
5: -0.1385660, 0.6695760, -0.1434896, 0.6777036, -0.8162696, 0.8130656
6: -0.2111069, 0.2549675, -0.2152857, 0.2608757, -0.4719827, 0.4702531
7: -0.3066055, 0.2155051, -0.3123749, 0.2203892, -0.5269948, 0.5278801
8: -0.2057540, 0.2902304, -0.2114335, 0.2966909, -0.5024449, 0.5016639
9: -0.3386965, 0.3081429, -0.3438668, 0.3164717, -0.6551682, 0.6520097

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.44 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
time: 1.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.10 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.10
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.10
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.10
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.10
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
time: 1.49 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.38 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.35 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
time: 2.11 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.24 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.35 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8132495, upper bound: 0.8062971
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1928342, 0.1904607, -0.2500582, 0.2443756, -0.4372098, 0.4405189
1: -0.1316828, 0.1445950, -0.1734074, 0.1877671, -0.3194499, 0.3180025
2: -0.1210749, 0.2188999, -0.1552624, 0.2720147, -0.3930897, 0.3741623
3: 0.2998670, 1.0480038, 0.1661023, 1.0579869, -0.7581198, 0.8819015
4: -0.1570106, 0.1462089, -0.2047127, 0.1891258, -0.3461364, 0.3509215
5: -0.0832145, 0.5591776, -0.1254750, 0.6633909, -0.7466055, 0.6846527
6: -0.1616564, 0.1887369, -0.2023995, 0.2440670, -0.4057235, 0.3911364
7: -0.2378954, 0.1618498, -0.2983487, 0.2024147, -0.4403101, 0.4601985
8: -0.1405334, 0.2173305, -0.1971189, 0.2782628, -0.4187962, 0.4144494
9: -0.2729709, 0.2185826, -0.3337545, 0.2865764, -0.5595474, 0.5523371

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 2.67 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1700392, 0.1697491, -0.2342213, 0.2286133, -0.3986525, 0.4039704
1: -0.1168623, 0.1270175, -0.1611305, 0.1758859, -0.2927482, 0.2881480
2: -0.1095840, 0.1973391, -0.1448576, 0.2573919, -0.3669759, 0.3421967
3: 0.3535259, 1.0441525, 0.2026195, 1.0553107, -0.7017848, 0.8415330
4: -0.1376660, 0.1310732, -0.1912944, 0.1761186, -0.3137846, 0.3223676
5: -0.0681767, 0.5157401, -0.1133615, 0.6348726, -0.7030493, 0.6291016
6: -0.1447474, 0.1678536, -0.1912021, 0.2278842, -0.3726316, 0.3590557
7: -0.2137316, 0.1502388, -0.2811326, 0.1910351, -0.4047667, 0.4313714
8: -0.1186267, 0.1933837, -0.1805858, 0.2615081, -0.3801348, 0.3739695
9: -0.2485444, 0.1931556, -0.3172618, 0.2666535, -0.5151980, 0.5104175

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8035020, upper bound: 0.8057050
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034655, upper bound: 0.8034655
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7930738, upper bound: 0.7989386
time: 1.34 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7929285, upper bound: 0.7929254
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2028642, 0.2011256, -0.2500582, 0.2443756, -0.4472399, 0.4511839
1: -0.1432786, 0.1543447, -0.1734074, 0.1877671, -0.3310457, 0.3277521
2: -0.1300209, 0.2327995, -0.1552624, 0.2720147, -0.4020356, 0.3880619
3: 0.2828985, 1.0483516, 0.1661023, 1.0579869, -0.7750884, 0.8822494
4: -0.1714482, 0.1572951, -0.2047127, 0.1891258, -0.3605739, 0.3620077
5: -0.0961161, 0.5694689, -0.1254750, 0.6633909, -0.7595071, 0.6949439
6: -0.1714040, 0.1990226, -0.2023995, 0.2440670, -0.4154710, 0.4014221
7: -0.2463606, 0.1755461, -0.2983487, 0.2024147, -0.4487753, 0.4738948
8: -0.1487307, 0.2303757, -0.1971189, 0.2782628, -0.4269935, 0.4274945
9: -0.2799888, 0.2392697, -0.3337545, 0.2865764, -0.5665653, 0.5730242

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1698477, 0.1738739, -0.2342213, 0.2286133, -0.3984610, 0.4080952
1: -0.1215079, 0.1292673, -0.1611305, 0.1758859, -0.2973938, 0.2903978
2: -0.1132401, 0.2025333, -0.1448576, 0.2573919, -0.3706320, 0.3473909
3: 0.3614141, 1.0416410, 0.2026195, 1.0553107, -0.6938966, 0.8390215
4: -0.1453356, 0.1350990, -0.1912944, 0.1761186, -0.3214542, 0.3263935
5: -0.0741304, 0.5064704, -0.1133615, 0.6348726, -0.7090030, 0.6198319
6: -0.1478697, 0.1689372, -0.1912021, 0.2278842, -0.3757539, 0.3601393
7: -0.2114801, 0.1550510, -0.2811326, 0.1910351, -0.4025152, 0.4361835
8: -0.1174472, 0.1957375, -0.1805858, 0.2615081, -0.3789552, 0.3763233
9: -0.2433822, 0.2037327, -0.3172618, 0.2666535, -0.5100358, 0.5209945

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8035020, upper bound: 0.8057857
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8034655, upper bound: 0.8034740
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2028642, 0.2011256, -0.2581915, 0.2553524, -0.4582167, 0.4593171
1: -0.1432786, 0.1543447, -0.1858926, 0.1966751, -0.3399537, 0.3402373
2: -0.1300209, 0.2327995, -0.1658475, 0.2849444, -0.4149652, 0.3986470
3: 0.2828985, 1.0483516, 0.1543276, 1.0586346, -0.7757362, 0.8940241
4: -0.1714482, 0.1572951, -0.2184964, 0.2024755, -0.3739237, 0.3757915
5: -0.0961161, 0.5694689, -0.1385660, 0.6695760, -0.7656921, 0.7080349
6: -0.1714040, 0.1990226, -0.2111069, 0.2549675, -0.4263715, 0.4101295
7: -0.2463606, 0.1755461, -0.3066055, 0.2155051, -0.4618657, 0.4821517
8: -0.1487307, 0.2303757, -0.2057540, 0.2902304, -0.4389611, 0.4361297
9: -0.2799888, 0.2392697, -0.3386965, 0.3081429, -0.5881317, 0.5779662

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1698477, 0.1738739, -0.2411986, 0.2385994, -0.4084471, 0.4150725
1: -0.1215079, 0.1292673, -0.1729589, 0.1840058, -0.3055137, 0.3022262
2: -0.1132401, 0.2025333, -0.1549075, 0.2693508, -0.3825909, 0.3574409
3: 0.3614141, 1.0416410, 0.1934627, 1.0555109, -0.6940967, 0.8481783
4: -0.1453356, 0.1350990, -0.2044517, 0.1887232, -0.3340588, 0.3395508
5: -0.0741304, 0.5064704, -0.1259177, 0.6388491, -0.7129794, 0.6323881
6: -0.1478697, 0.1689372, -0.1992192, 0.2376846, -0.3855543, 0.3681564
7: -0.2114801, 0.1550510, -0.2880291, 0.2035665, -0.4150466, 0.4430801
8: -0.1174472, 0.1957375, -0.1880914, 0.2722728, -0.3897199, 0.3838289
9: -0.2433822, 0.2037327, -0.3208368, 0.2871346, -0.5305168, 0.5245694

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7930738, upper bound: 0.7990476
time: 1.50 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7929254, upper bound: 0.7929285
time: 1.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.26 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8035020, upper bound: 0.8057050
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8034655, upper bound: 0.8034655
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.7930738, upper bound: 0.7989386
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.7929285, upper bound: 0.7929254
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8035020, upper bound: 0.8057857
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8034655, upper bound: 0.8034740
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.8060520, upper bound: 0.8060520
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.7930738, upper bound: 0.7990476
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 3, lower bound: -0.7929254, upper bound: 0.7929285

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1928342, 0.1904607, -0.1928342, 0.1904607, -0.3832949, 0.3832949
1: -0.1316828, 0.1445950, -0.1316828, 0.1445950, -0.2762779, 0.2762779
2: -0.1210749, 0.2188999, -0.1210749, 0.2188999, -0.3399748, 0.3399748
3: 0.2998670, 1.0480038, 0.2998670, 1.0480038, -0.7481368, 0.7481368
4: -0.1570106, 0.1462089, -0.1570106, 0.1462089, -0.3032195, 0.3032195
5: -0.0832145, 0.5591776, -0.0832145, 0.5591776, -0.6423922, 0.6423922
6: -0.1616564, 0.1887369, -0.1616564, 0.1887369, -0.3503933, 0.3503933
7: -0.2378954, 0.1618498, -0.2378954, 0.1618498, -0.3997451, 0.3997451
8: -0.1405334, 0.2173305, -0.1405334, 0.2173305, -0.3578639, 0.3578639
9: -0.2729709, 0.2185826, -0.2729709, 0.2185826, -0.4915535, 0.4915535

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1928342, 0.1904607, -0.1700392, 0.1697491, -0.3625833, 0.3604999
1: -0.1316828, 0.1445950, -0.1168623, 0.1270175, -0.2587004, 0.2614574
2: -0.1210749, 0.2188999, -0.1095840, 0.1973391, -0.3184140, 0.3284839
3: 0.2998670, 1.0480038, 0.3535259, 1.0441525, -0.7442855, 0.6944779
4: -0.1570106, 0.1462089, -0.1376660, 0.1310732, -0.2880839, 0.2838749
5: -0.0832145, 0.5591776, -0.0681767, 0.5157401, -0.5989546, 0.6273543
6: -0.1616564, 0.1887369, -0.1447474, 0.1678536, -0.3295100, 0.3334844
7: -0.2378954, 0.1618498, -0.2137316, 0.1502388, -0.3881342, 0.3755814
8: -0.1405334, 0.2173305, -0.1186267, 0.1933837, -0.3339171, 0.3359572
9: -0.2729709, 0.2185826, -0.2485444, 0.1931556, -0.4661266, 0.4671270

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1690943, 0.1688543, -0.2226183, 0.2166913, -0.3857855, 0.3914725
1: -0.1161665, 0.1263258, -0.1505585, 0.1669702, -0.2831367, 0.2768843
2: -0.1090239, 0.1963520, -0.1358902, 0.2451377, -0.3541616, 0.3322422
3: 0.3554516, 1.0439562, 0.2258523, 1.0527583, -0.6973068, 0.8181038
4: -0.1367585, 0.1304035, -0.1794716, 0.1654891, -0.3022476, 0.3098751
5: -0.0674736, 0.5143384, -0.1025296, 0.6179978, -0.6854714, 0.6168680
6: -0.1440174, 0.1669702, -0.1822927, 0.2157518, -0.3597691, 0.3492629
7: -0.2128357, 0.1496319, -0.2694196, 0.1803909, -0.3932266, 0.4190515
8: -0.1177568, 0.1922946, -0.1688992, 0.2478859, -0.3656428, 0.3611938
9: -0.2476755, 0.1918290, -0.3064381, 0.2488165, -0.4964920, 0.4982671

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928173
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1691459, 0.1688914, -0.2506177, 0.2448889, -0.4140348, 0.4195092
1: -0.1161886, 0.1263411, -0.1741652, 0.1883706, -0.3045591, 0.3005064
2: -0.1090427, 0.1963883, -0.1558241, 0.2726204, -0.3816630, 0.3522125
3: 0.3553681, 1.0439856, 0.1661890, 1.0572103, -0.7018422, 0.8777966
4: -0.1367772, 0.1304225, -0.2056279, 0.1898767, -0.3266539, 0.3360504
5: -0.0674908, 0.5143963, -0.1263336, 0.6648675, -0.7323583, 0.6407300
6: -0.1440408, 0.1670114, -0.2029888, 0.2445784, -0.3886192, 0.3700002
7: -0.2128776, 0.1496584, -0.2987874, 0.2031641, -0.4160417, 0.4484458
8: -0.1178063, 0.1923494, -0.1974593, 0.2785859, -0.3963922, 0.3898087
9: -0.2477298, 0.1918811, -0.3342946, 0.2861739, -0.5339038, 0.5261758

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908337
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1928342, 0.1904607, -0.2028642, 0.2011256, -0.3939599, 0.3933249
1: -0.1316828, 0.1445950, -0.1432786, 0.1543447, -0.2860276, 0.2878737
2: -0.1210749, 0.2188999, -0.1300209, 0.2327995, -0.3538744, 0.3489208
3: 0.2998670, 1.0480038, 0.2828985, 1.0483516, -0.7484846, 0.7651053
4: -0.1570106, 0.1462089, -0.1714482, 0.1572951, -0.3143057, 0.3176570
5: -0.0832145, 0.5591776, -0.0961161, 0.5694689, -0.6526834, 0.6552937
6: -0.1616564, 0.1887369, -0.1714040, 0.1990226, -0.3606790, 0.3601409
7: -0.2378954, 0.1618498, -0.2463606, 0.1755461, -0.4134415, 0.4082103
8: -0.1405334, 0.2173305, -0.1487307, 0.2303757, -0.3709090, 0.3660612
9: -0.2729709, 0.2185826, -0.2799888, 0.2392697, -0.5122406, 0.4985714

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8058320, upper bound: 0.7933126
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1928342, 0.1904607, -0.1698477, 0.1738739, -0.3667081, 0.3603084
1: -0.1316828, 0.1445950, -0.1215079, 0.1292673, -0.2609501, 0.2661029
2: -0.1210749, 0.2188999, -0.1132401, 0.2025333, -0.3236082, 0.3321400
3: 0.2998670, 1.0480038, 0.3614141, 1.0416410, -0.7417740, 0.6865897
4: -0.1570106, 0.1462089, -0.1453356, 0.1350990, -0.2921097, 0.2915445
5: -0.0832145, 0.5591776, -0.0741304, 0.5064704, -0.5896849, 0.6333080
6: -0.1616564, 0.1887369, -0.1478697, 0.1689372, -0.3305936, 0.3366066
7: -0.2378954, 0.1618498, -0.2114801, 0.1550510, -0.3929463, 0.3733299
8: -0.1405334, 0.2173305, -0.1174472, 0.1957375, -0.3362709, 0.3347777
9: -0.2729709, 0.2185826, -0.2433822, 0.2037327, -0.4767036, 0.4619648

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8058320, upper bound: 0.7933126
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1700392, 0.1697491, -0.1823466, 0.1848758, -0.3549150, 0.3520957
1: -0.1168623, 0.1270175, -0.1308306, 0.1398219, -0.2566842, 0.2578481
2: -0.1095840, 0.1973391, -0.1201123, 0.2151396, -0.3247236, 0.3174514
3: 0.3535259, 1.0441525, 0.3331147, 1.0416272, -0.6881013, 0.7110378
4: -0.1376660, 0.1310732, -0.1565190, 0.1446831, -0.2823491, 0.2875922
5: -0.0681767, 0.5157401, -0.0833027, 0.5309573, -0.5991340, 0.5990428
6: -0.1447474, 0.1678536, -0.1577751, 0.1807420, -0.3254894, 0.3256288
7: -0.2137316, 0.1502388, -0.2243968, 0.1637465, -0.3774781, 0.3746356
8: -0.1186267, 0.1933837, -0.1283412, 0.2087233, -0.3273500, 0.3217249
9: -0.2485444, 0.1931556, -0.2566607, 0.2154681, -0.4640125, 0.4498163

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7929285, upper bound: 0.7929254
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7929285, upper bound: 0.7929254
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1559681, 0.1576106, -0.1536300, 0.1615272, -0.3174953, 0.3112406
1: -0.1077271, 0.1179539, -0.1111113, 0.1168928, -0.2246200, 0.2290652
2: -0.1031826, 0.1838737, -0.1048237, 0.1871168, -0.2902994, 0.2886974
3: 0.3858503, 1.0409738, 0.4005578, 1.0354383, -0.6495880, 0.6404160
4: -0.1265108, 0.1218139, -0.1310910, 0.1242782, -0.2507890, 0.2529050
5: -0.0600203, 0.4905151, -0.0625004, 0.4784718, -0.5384921, 0.5530155
6: -0.1347217, 0.1549277, -0.1371969, 0.1543912, -0.2891130, 0.2921247
7: -0.1996915, 0.1429581, -0.1940552, 0.1468231, -0.3465146, 0.3370133
8: -0.1051887, 0.1788850, -0.1009129, 0.1785775, -0.2837663, 0.2797979
9: -0.2339030, 0.1773981, -0.2261765, 0.1794366, -0.4133396, 0.4035746

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2028642, 0.2011256, -0.1928342, 0.1904607, -0.3933249, 0.3939599
1: -0.1432786, 0.1543447, -0.1316828, 0.1445950, -0.2878737, 0.2860276
2: -0.1300209, 0.2327995, -0.1210749, 0.2188999, -0.3489208, 0.3538744
3: 0.2828985, 1.0483516, 0.2998670, 1.0480038, -0.7651053, 0.7484846
4: -0.1714482, 0.1572951, -0.1570106, 0.1462089, -0.3176570, 0.3143057
5: -0.0961161, 0.5694689, -0.0832145, 0.5591776, -0.6552937, 0.6526834
6: -0.1714040, 0.1990226, -0.1616564, 0.1887369, -0.3601409, 0.3606790
7: -0.2463606, 0.1755461, -0.2378954, 0.1618498, -0.4082103, 0.4134415
8: -0.1487307, 0.2303757, -0.1405334, 0.2173305, -0.3660612, 0.3709090
9: -0.2799888, 0.2392697, -0.2729709, 0.2185826, -0.4985714, 0.5122406

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
time: 1.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2028642, 0.2011256, -0.1700392, 0.1697491, -0.3726133, 0.3711649
1: -0.1432786, 0.1543447, -0.1168623, 0.1270175, -0.2702962, 0.2712070
2: -0.1300209, 0.2327995, -0.1095840, 0.1973391, -0.3273600, 0.3423835
3: 0.2828985, 1.0483516, 0.3535259, 1.0441525, -0.7612540, 0.6948258
4: -0.1714482, 0.1572951, -0.1376660, 0.1310732, -0.3025213, 0.2949612
5: -0.0961161, 0.5694689, -0.0681767, 0.5157401, -0.6118562, 0.6376455
6: -0.1714040, 0.1990226, -0.1447474, 0.1678536, -0.3392576, 0.3437700
7: -0.2463606, 0.1755461, -0.2137316, 0.1502388, -0.3965994, 0.3892778
8: -0.1487307, 0.2303757, -0.1186267, 0.1933837, -0.3421143, 0.3490024
9: -0.2799888, 0.2392697, -0.2485444, 0.1931556, -0.4731444, 0.4878141

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1689430, 0.1730888, -0.2226183, 0.2166913, -0.3856342, 0.3957070
1: -0.1208582, 0.1286111, -0.1505585, 0.1669702, -0.2878284, 0.2791696
2: -0.1127163, 0.2015955, -0.1358902, 0.2451377, -0.3578540, 0.3374857
3: 0.3632870, 1.0414387, 0.2258523, 1.0527583, -0.6894713, 0.8155864
4: -0.1444984, 0.1344669, -0.1794716, 0.1654891, -0.3099875, 0.3139384
5: -0.0734852, 0.5051298, -0.1025296, 0.6179978, -0.6914829, 0.6076593
6: -0.1471806, 0.1680979, -0.1822927, 0.2157518, -0.3629324, 0.3503906
7: -0.2106167, 0.1544827, -0.2694196, 0.1803909, -0.3910076, 0.4239023
8: -0.1166066, 0.1946964, -0.1688992, 0.2478859, -0.3644926, 0.3635957
9: -0.2425461, 0.2024888, -0.3064381, 0.2488165, -0.4913626, 0.5089270

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928726
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1690691, 0.1731782, -0.2506177, 0.2448889, -0.4139580, 0.4237959
1: -0.1209307, 0.1286818, -0.1741652, 0.1883706, -0.3093012, 0.3028470
2: -0.1127760, 0.2017066, -0.1558241, 0.2726204, -0.3853964, 0.3575307
3: 0.3630274, 1.0414813, 0.1661890, 1.0572103, -0.6941830, 0.8752923
4: -0.1445775, 0.1345352, -0.2056279, 0.1898767, -0.3344542, 0.3401631
5: -0.0735467, 0.5053164, -0.1263336, 0.6648675, -0.7384142, 0.6316500
6: -0.1472608, 0.1682087, -0.2029888, 0.2445784, -0.3918392, 0.3711975
7: -0.2107328, 0.1545542, -0.2987874, 0.2031641, -0.4138969, 0.4533415
8: -0.1167229, 0.1948354, -0.1974593, 0.2785859, -0.3953089, 0.3922946
9: -0.2426726, 0.2026413, -0.3342946, 0.2861739, -0.5288465, 0.5369359

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908369
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2028642, 0.2011256, -0.2028642, 0.2011256, -0.4039899, 0.4039899
1: -0.1432786, 0.1543447, -0.1432786, 0.1543447, -0.2976233, 0.2976233
2: -0.1300209, 0.2327995, -0.1300209, 0.2327995, -0.3628204, 0.3628204
3: 0.2828985, 1.0483516, 0.2828985, 1.0483516, -0.7654532, 0.7654532
4: -0.1714482, 0.1572951, -0.1714482, 0.1572951, -0.3287432, 0.3287432
5: -0.0961161, 0.5694689, -0.0961161, 0.5694689, -0.6655849, 0.6655849
6: -0.1714040, 0.1990226, -0.1714040, 0.1990226, -0.3704266, 0.3704266
7: -0.2463606, 0.1755461, -0.2463606, 0.1755461, -0.4219067, 0.4219067
8: -0.1487307, 0.2303757, -0.1487307, 0.2303757, -0.3791063, 0.3791063
9: -0.2799888, 0.2392697, -0.2799888, 0.2392697, -0.5192585, 0.5192585

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
time: 1.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 2.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2028642, 0.2011256, -0.1698477, 0.1738739, -0.3767381, 0.3709733
1: -0.1432786, 0.1543447, -0.1215079, 0.1292673, -0.2725459, 0.2758526
2: -0.1300209, 0.2327995, -0.1132401, 0.2025333, -0.3325542, 0.3460396
3: 0.2828985, 1.0483516, 0.3614141, 1.0416410, -0.7587425, 0.6869375
4: -0.1714482, 0.1572951, -0.1453356, 0.1350990, -0.3065472, 0.3026307
5: -0.0961161, 0.5694689, -0.0741304, 0.5064704, -0.6025865, 0.6435992
6: -0.1714040, 0.1990226, -0.1478697, 0.1689372, -0.3403412, 0.3468923
7: -0.2463606, 0.1755461, -0.2114801, 0.1550510, -0.4014116, 0.3870263
8: -0.1487307, 0.2303757, -0.1174472, 0.1957375, -0.3444682, 0.3478228
9: -0.2799888, 0.2392697, -0.2433822, 0.2037327, -0.4837215, 0.4826519

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1698477, 0.1738739, -0.1823466, 0.1848758, -0.3547235, 0.3562205
1: -0.1215079, 0.1292673, -0.1308306, 0.1398219, -0.2613298, 0.2600979
2: -0.1132401, 0.2025333, -0.1201123, 0.2151396, -0.3283797, 0.3226456
3: 0.3614141, 1.0416410, 0.3331147, 1.0416272, -0.6802130, 0.7085263
4: -0.1453356, 0.1350990, -0.1565190, 0.1446831, -0.2900187, 0.2916180
5: -0.0741304, 0.5064704, -0.0833027, 0.5309573, -0.6050876, 0.5897731
6: -0.1478697, 0.1689372, -0.1577751, 0.1807420, -0.3286117, 0.3267123
7: -0.2114801, 0.1550510, -0.2243968, 0.1637465, -0.3752266, 0.3794478
8: -0.1174472, 0.1957375, -0.1283412, 0.2087233, -0.3261704, 0.3240787
9: -0.2433822, 0.2037327, -0.2566607, 0.2154681, -0.4588503, 0.4603934

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7929254, upper bound: 0.7929285
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7929254, upper bound: 0.7929285
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1546687, 0.1615878, -0.1536300, 0.1615272, -0.3161959, 0.3152177
1: -0.1115133, 0.1177644, -0.1111113, 0.1168928, -0.2284061, 0.2288757
2: -0.1052911, 0.1879223, -0.1048237, 0.1871168, -0.2924079, 0.2927460
3: 0.3971850, 1.0379288, 0.4005578, 1.0354383, -0.6382533, 0.6373710
4: -0.1324655, 0.1249762, -0.1310910, 0.1242782, -0.2567438, 0.2560672
5: -0.0641854, 0.4792956, -0.0625004, 0.4784718, -0.5426572, 0.5417960
6: -0.1374239, 0.1550166, -0.1371969, 0.1543912, -0.2918151, 0.2922136
7: -0.1956195, 0.1471029, -0.1940552, 0.1468231, -0.3424427, 0.3411582
8: -0.1029415, 0.1791309, -0.1009129, 0.1785775, -0.2815190, 0.2800438
9: -0.2276118, 0.1849582, -0.2261765, 0.1794366, -0.4070484, 0.4111347

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7926662, upper bound: 0.7910332
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909993, upper bound: 0.7910107
time: 1.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.33 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928173
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908337
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8058320, upper bound: 0.7933126
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8058320, upper bound: 0.7933126
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7929285, upper bound: 0.7929254
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7929285, upper bound: 0.7929254
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928726
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908369
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8051167, upper bound: 0.7933126
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7929254, upper bound: 0.7929285
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7929254, upper bound: 0.7929285
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7926662, upper bound: 0.7910332
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.33
Output dim: 3, lower bound: -0.7909993, upper bound: 0.7910107

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1928342, 0.1904607, -0.3228142, 0.3326481
1: -0.0939614, 0.1055578, -0.1316828, 0.1445950, -0.2385564, 0.2372407
2: -0.0941783, 0.1624106, -0.1210749, 0.2188999, -0.3130782, 0.2834855
3: 0.4443746, 1.0336604, 0.2998670, 1.0480038, -0.6036292, 0.7337934
4: -0.1111998, 0.1083961, -0.1570106, 0.1462089, -0.2574086, 0.2654067
5: -0.0505697, 0.4460281, -0.0832145, 0.5591776, -0.6097474, 0.5292426
6: -0.1205043, 0.1328021, -0.1616564, 0.1887369, -0.3092412, 0.2944584
7: -0.1758752, 0.1307725, -0.2378954, 0.1618498, -0.3377250, 0.3686679
8: -0.0818439, 0.1581148, -0.1405334, 0.2173305, -0.2991744, 0.2986482
9: -0.2079326, 0.1538350, -0.2729709, 0.2185826, -0.4265152, 0.4268059

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 1.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1186917, 0.1269170, -0.1787107, 0.1774390, -0.2961306, 0.3056277
1: -0.0852304, 0.0961525, -0.1223416, 0.1340099, -0.2192403, 0.2184941
2: -0.0878181, 0.1474300, -0.1136751, 0.2054819, -0.2933000, 0.2611051
3: 0.4772729, 1.0306065, 0.3333406, 1.0449959, -0.5677230, 0.6972659
4: -0.1003226, 0.0999454, -0.1450791, 0.1368791, -0.2372017, 0.2450245
5: -0.0457663, 0.4164358, -0.0737160, 0.5334092, -0.5791755, 0.4901518
6: -0.1106552, 0.1185563, -0.1512985, 0.1756778, -0.2863331, 0.2698548
7: -0.1618960, 0.1219839, -0.2231492, 0.1538145, -0.3157105, 0.3451331
8: -0.0691035, 0.1477058, -0.1270452, 0.2019821, -0.2710856, 0.2747510
9: -0.1941239, 0.1472560, -0.2579537, 0.2013957, -0.3955196, 0.4052097

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013185
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988673
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1700392, 0.1697491, -0.3021026, 0.3098531
1: -0.0939614, 0.1055578, -0.1168623, 0.1270175, -0.2209789, 0.2224202
2: -0.0941783, 0.1624106, -0.1095840, 0.1973391, -0.2915174, 0.2719946
3: 0.4443746, 1.0336604, 0.3535259, 1.0441525, -0.5997779, 0.6801345
4: -0.1111998, 0.1083961, -0.1376660, 0.1310732, -0.2422730, 0.2460621
5: -0.0505697, 0.4460281, -0.0681767, 0.5157401, -0.5663098, 0.5142048
6: -0.1205043, 0.1328021, -0.1447474, 0.1678536, -0.2883579, 0.2775495
7: -0.1758752, 0.1307725, -0.2137316, 0.1502388, -0.3261141, 0.3445042
8: -0.0818439, 0.1581148, -0.1186267, 0.1933837, -0.2752276, 0.2767415
9: -0.2079326, 0.1538350, -0.2485444, 0.1931556, -0.4010882, 0.4023795

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1186917, 0.1269170, -0.1559681, 0.1576106, -0.2763023, 0.2828851
1: -0.0852304, 0.0961525, -0.1077271, 0.1179539, -0.2031844, 0.2038796
2: -0.0878181, 0.1474300, -0.1031826, 0.1838737, -0.2716917, 0.2506126
3: 0.4772729, 1.0306065, 0.3858503, 1.0409738, -0.5637009, 0.6447563
4: -0.1003226, 0.0999454, -0.1265108, 0.1218139, -0.2221365, 0.2264561
5: -0.0457663, 0.4164358, -0.0600203, 0.4905151, -0.5362814, 0.4764561
6: -0.1106552, 0.1185563, -0.1347217, 0.1549277, -0.2655830, 0.2532780
7: -0.1618960, 0.1219839, -0.1996915, 0.1429581, -0.3048541, 0.3216754
8: -0.0691035, 0.1477058, -0.1051887, 0.1788850, -0.2479885, 0.2528945
9: -0.1941239, 0.1472560, -0.2339030, 0.1773981, -0.3715220, 0.3811590

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7977373, upper bound: 0.7929228
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909334
time: 1.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1124734, 0.1212652, -0.2226183, 0.2166913, -0.3291647, 0.3438835
1: -0.0814218, 0.0917020, -0.1505585, 0.1669702, -0.2483919, 0.2422605
2: -0.0848431, 0.1414255, -0.1358902, 0.2451377, -0.3299807, 0.2773157
3: 0.4962844, 1.0292321, 0.2258523, 1.0527583, -0.5564739, 0.8033798
4: -0.0966727, 0.0968332, -0.1794716, 0.1654891, -0.2621618, 0.2763048
5: -0.0441654, 0.4002295, -0.1025296, 0.6179978, -0.6621632, 0.5027591
6: -0.1070608, 0.1117572, -0.1822927, 0.2157518, -0.3228126, 0.2940499
7: -0.1540299, 0.1187557, -0.2694196, 0.1803909, -0.3344207, 0.3881752
8: -0.0621860, 0.1440119, -0.1688992, 0.2478859, -0.3100719, 0.3129111
9: -0.1875473, 0.1442345, -0.3064381, 0.2488165, -0.4363638, 0.4506726

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 2.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1002065, 0.1086552, -0.2081929, 0.2038512, -0.3040577, 0.3168481
1: -0.0733034, 0.0826874, -0.1405014, 0.1561741, -0.2294775, 0.2231888
2: -0.0786476, 0.1291508, -0.1277822, 0.2318663, -0.3105139, 0.2569331
3: 0.5287691, 1.0265702, 0.2588924, 1.0498757, -0.5211066, 0.7676778
4: -0.0863403, 0.0885295, -0.1672998, 0.1555107, -0.2418509, 0.2558293
5: -0.0394679, 0.3723960, -0.0918130, 0.5923262, -0.6317941, 0.4642090
6: -0.0975457, 0.0982279, -0.1721295, 0.2026924, -0.3002381, 0.2703574
7: -0.1404083, 0.1101410, -0.2547638, 0.1702741, -0.3106824, 0.3649048
8: -0.0509891, 0.1339294, -0.1552671, 0.2326665, -0.2836556, 0.2891966
9: -0.1745202, 0.1375674, -0.2912222, 0.2318400, -0.4063602, 0.4287896

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1124999, 0.1212799, -0.2506177, 0.2448889, -0.3573889, 0.3718977
1: -0.0814334, 0.0917082, -0.1741652, 0.1883706, -0.2698039, 0.2658734
2: -0.0848542, 0.1414485, -0.1558241, 0.2726204, -0.3574746, 0.2972726
3: 0.4962292, 1.0292531, 0.1661890, 1.0572103, -0.5609812, 0.8630642
4: -0.0966792, 0.0968388, -0.2056279, 0.1898767, -0.2865559, 0.3024667
5: -0.0441699, 0.4002717, -0.1263336, 0.6648675, -0.7090373, 0.5266053
6: -0.1070660, 0.1117856, -0.2029888, 0.2445784, -0.3516445, 0.3147744
7: -0.1540556, 0.1187765, -0.2987874, 0.2031641, -0.3572197, 0.4175638
8: -0.0622183, 0.1440273, -0.1974593, 0.2785859, -0.3408042, 0.3414866
9: -0.1875719, 0.1442588, -0.3342946, 0.2861739, -0.4737459, 0.4785534

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1002313, 0.1086699, -0.2356418, 0.2299536, -0.3301849, 0.3443117
1: -0.0733144, 0.0826953, -0.1627637, 0.1772753, -0.2505898, 0.2454590
2: -0.0786585, 0.1291733, -0.1461354, 0.2589366, -0.3375951, 0.2753087
3: 0.5287286, 1.0265849, 0.2011491, 1.0542927, -0.5255641, 0.8254358
4: -0.0863500, 0.0885365, -0.1930673, 0.1777994, -0.2641493, 0.2816038
5: -0.0394732, 0.3724322, -0.1150068, 0.6375265, -0.6769997, 0.4874390
6: -0.0975534, 0.0982533, -0.1924996, 0.2293131, -0.3268666, 0.2907529
7: -0.1404310, 0.1101577, -0.2823298, 0.1927167, -0.3331477, 0.3924875
8: -0.0510122, 0.1339464, -0.1816034, 0.2627561, -0.3137683, 0.3155498
9: -0.1745385, 0.1375798, -0.3184251, 0.2669476, -0.4414861, 0.4560049

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.2028642, 0.2011256, -0.3334792, 0.3426782
1: -0.0939614, 0.1055578, -0.1432786, 0.1543447, -0.2483061, 0.2488365
2: -0.0941783, 0.1624106, -0.1300209, 0.2327995, -0.3269778, 0.2924315
3: 0.4443746, 1.0336604, 0.2828985, 1.0483516, -0.6039771, 0.7507619
4: -0.1111998, 0.1083961, -0.1714482, 0.1572951, -0.2684948, 0.2798442
5: -0.0505697, 0.4460281, -0.0961161, 0.5694689, -0.6200386, 0.5421442
6: -0.1205043, 0.1328021, -0.1714040, 0.1990226, -0.3195269, 0.3042061
7: -0.1758752, 0.1307725, -0.2463606, 0.1755461, -0.3514214, 0.3771331
8: -0.0818439, 0.1581148, -0.1487307, 0.2303757, -0.3122196, 0.3068455
9: -0.2079326, 0.1538350, -0.2799888, 0.2392697, -0.4472023, 0.4338238

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1186917, 0.1269170, -0.1870677, 0.1868715, -0.3055632, 0.3139847
1: -0.0852304, 0.0961525, -0.1323955, 0.1422783, -0.2275088, 0.2285479
2: -0.0878181, 0.1474300, -0.1214969, 0.2179812, -0.3057992, 0.2689269
3: 0.4772729, 1.0306065, 0.3200509, 1.0447118, -0.5674390, 0.7105557
4: -0.1003226, 0.0999454, -0.1583137, 0.1463817, -0.2467043, 0.2582590
5: -0.0457663, 0.4164358, -0.0848192, 0.5408301, -0.5865964, 0.5012550
6: -0.1106552, 0.1185563, -0.1601073, 0.1845536, -0.2952088, 0.2786635
7: -0.1618960, 0.1219839, -0.2298687, 0.1646079, -0.3265040, 0.3518525
8: -0.0691035, 0.1477058, -0.1336126, 0.2133581, -0.2824616, 0.2813184
9: -0.1941239, 0.1472560, -0.2628790, 0.2206163, -0.4147403, 0.4101350

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013185
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988727, upper bound: 0.7988673
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1698477, 0.1738739, -0.3062274, 0.3096616
1: -0.0939614, 0.1055578, -0.1215079, 0.1292673, -0.2232287, 0.2270657
2: -0.0941783, 0.1624106, -0.1132401, 0.2025333, -0.2967117, 0.2756507
3: 0.4443746, 1.0336604, 0.3614141, 1.0416410, -0.5972664, 0.6722463
4: -0.1111998, 0.1083961, -0.1453356, 0.1350990, -0.2462988, 0.2537317
5: -0.0505697, 0.4460281, -0.0741304, 0.5064704, -0.5570401, 0.5201585
6: -0.1205043, 0.1328021, -0.1478697, 0.1689372, -0.2894415, 0.2806718
7: -0.1758752, 0.1307725, -0.2114801, 0.1550510, -0.3309262, 0.3422527
8: -0.0818439, 0.1581148, -0.1174472, 0.1957375, -0.2775815, 0.2755619
9: -0.2079326, 0.1538350, -0.2433822, 0.2037327, -0.4116653, 0.3972173

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1186917, 0.1269170, -0.1546687, 0.1615878, -0.2802795, 0.2815856
1: -0.0852304, 0.0961525, -0.1115133, 0.1177644, -0.2029949, 0.2076658
2: -0.0878181, 0.1474300, -0.1052911, 0.1879223, -0.2757404, 0.2527211
3: 0.4772729, 1.0306065, 0.3971850, 1.0379288, -0.5606560, 0.6334215
4: -0.1003226, 0.0999454, -0.1324655, 0.1249762, -0.2252988, 0.2324109
5: -0.0457663, 0.4164358, -0.0641854, 0.4792956, -0.5250619, 0.4806212
6: -0.1106552, 0.1185563, -0.1374239, 0.1550166, -0.2656719, 0.2559801
7: -0.1618960, 0.1219839, -0.1956195, 0.1471029, -0.3089989, 0.3176034
8: -0.0691035, 0.1477058, -0.1029415, 0.1791309, -0.2482344, 0.2506472
9: -0.1941239, 0.1472560, -0.2276118, 0.1849582, -0.3790821, 0.3748678

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7979024, upper bound: 0.7929228
time: 1.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978998, upper bound: 0.7912154
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1131819, 0.1219674, -0.1823466, 0.1848758, -0.2980577, 0.3043140
1: -0.0819143, 0.0920996, -0.1308306, 0.1398219, -0.2217361, 0.2229302
2: -0.0852243, 0.1422078, -0.1201123, 0.2151396, -0.3003639, 0.2623201
3: 0.4948269, 1.0293642, 0.3331147, 1.0416272, -0.5468003, 0.6962495
4: -0.0972459, 0.0972678, -0.1565190, 0.1446831, -0.2419290, 0.2537868
5: -0.0444142, 0.4013967, -0.0833027, 0.5309573, -0.5753714, 0.4846994
6: -0.1075819, 0.1124838, -0.1577751, 0.1807420, -0.2883239, 0.2702590
7: -0.1546251, 0.1193287, -0.2243968, 0.1637465, -0.3183716, 0.3437256
8: -0.0627786, 0.1446541, -0.1283412, 0.2087233, -0.2715019, 0.2729953
9: -0.1881767, 0.1446504, -0.2566607, 0.2154681, -0.4036447, 0.4013111

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908841, upper bound: 0.7982595
time: 1.44 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1008312, 0.1093591, -0.1823466, 0.1848758, -0.2857070, 0.2917057
1: -0.0737615, 0.0830829, -0.1308306, 0.1398219, -0.2135834, 0.2139135
2: -0.0790291, 0.1297382, -0.1201123, 0.2151396, -0.2941687, 0.2498505
3: 0.5273200, 1.0266857, 0.3331147, 1.0416272, -0.5143071, 0.6935710
4: -0.0868989, 0.0889649, -0.1565190, 0.1446831, -0.2315820, 0.2454839
5: -0.0397120, 0.3735656, -0.0833027, 0.5309573, -0.5706693, 0.4568684
6: -0.0980690, 0.0989287, -0.1577751, 0.1807420, -0.2788109, 0.2567039
7: -0.1409962, 0.1107113, -0.2243968, 0.1637465, -0.3047427, 0.3351081
8: -0.0514807, 0.1345519, -0.1283412, 0.2087233, -0.2602040, 0.2628931
9: -0.1750948, 0.1379746, -0.2566607, 0.2154681, -0.3905628, 0.3946353

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908841, upper bound: 0.7982595
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1448753, 0.1472884, -0.1527154, 0.1607351, -0.3056104, 0.3000038
1: -0.0990862, 0.1126179, -0.1104556, 0.1163444, -0.2154306, 0.2230735
2: -0.0980847, 0.1718817, -0.1043601, 0.1861702, -0.2842549, 0.2762418
3: 0.4069002, 1.0387719, 0.4023970, 1.0352353, -0.6283351, 0.6363748
4: -0.1165836, 0.1136026, -0.1303014, 0.1236413, -0.2402249, 0.2439040
5: -0.0532438, 0.4745703, -0.0619224, 0.4771147, -0.5303584, 0.5364927
6: -0.1264503, 0.1444066, -0.1365613, 0.1535418, -0.2799920, 0.2809679
7: -0.1904205, 0.1353994, -0.1932238, 0.1462515, -0.3366720, 0.3286232
8: -0.0949170, 0.1676144, -0.1000599, 0.1777478, -0.2726648, 0.2676743
9: -0.2240686, 0.1639596, -0.2253275, 0.1782827, -0.4023513, 0.3892871

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1683605, 0.1682937, -0.1528455, 0.1608297, -0.3291902, 0.3211392
1: -0.1163364, 0.1248079, -0.1105327, 0.1163962, -0.2327327, 0.2353406
2: -0.1081156, 0.1964567, -0.1044118, 0.1862868, -0.2944024, 0.3008685
3: 0.3592461, 1.0426173, 0.4021413, 1.0352788, -0.6760327, 0.6404760
4: -0.1365195, 0.1304664, -0.1303765, 0.1237125, -0.2602320, 0.2608429
5: -0.0667525, 0.5125632, -0.0619751, 0.4773046, -0.5440571, 0.5745383
6: -0.1437285, 0.1665697, -0.1366250, 0.1536577, -0.2973862, 0.3031947
7: -0.2112145, 0.1497268, -0.1933362, 0.1463276, -0.3575420, 0.3430630
8: -0.1165961, 0.1910026, -0.1001812, 0.1778360, -0.2944321, 0.2911838
9: -0.2464552, 0.1890608, -0.2254584, 0.1784256, -0.4248808, 0.4145192

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.1928342, 0.1904607, -0.3358466, 0.3470003
1: -0.1052670, 0.1127120, -0.1316828, 0.1445950, -0.2498620, 0.2443948
2: -0.1008987, 0.1787496, -0.1210749, 0.2188999, -0.3197986, 0.2998245
3: 0.4196393, 1.0340916, 0.2998670, 1.0480038, -0.6283645, 0.7342246
4: -0.1251058, 0.1190936, -0.1570106, 0.1462089, -0.2713146, 0.2761042
5: -0.0586379, 0.4646062, -0.0832145, 0.5591776, -0.6178155, 0.5478207
6: -0.1318853, 0.1461769, -0.1616564, 0.1887369, -0.3206223, 0.3078333
7: -0.1864930, 0.1413886, -0.2378954, 0.1618498, -0.3483428, 0.3792840
8: -0.0932197, 0.1716688, -0.1405334, 0.2173305, -0.3105502, 0.3122022
9: -0.2185586, 0.1716712, -0.2729709, 0.2185826, -0.4371412, 0.4446421

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 1.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1207468, 0.1308932, -0.1787107, 0.1774390, -0.2981858, 0.3096040
1: -0.0882468, 0.0960897, -0.1223416, 0.1340099, -0.2222567, 0.2184313
2: -0.0899592, 0.1514581, -0.1136751, 0.2054819, -0.2954411, 0.2651332
3: 0.4851008, 1.0281571, 0.3333406, 1.0449959, -0.5598950, 0.6948165
4: -0.1048785, 0.1029995, -0.1450791, 0.1368791, -0.2417576, 0.2480786
5: -0.0474810, 0.4096504, -0.0737160, 0.5334092, -0.5808902, 0.4833664
6: -0.1147075, 0.1198550, -0.1512985, 0.1756778, -0.2903853, 0.2711535
7: -0.1586763, 0.1267411, -0.2231492, 0.1538145, -0.3124908, 0.3498904
8: -0.0671329, 0.1537934, -0.1270452, 0.2019821, -0.2691150, 0.2808386
9: -0.1936240, 0.1493764, -0.2579537, 0.2013957, -0.3950197, 0.4073302

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013759
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.1700392, 0.1697491, -0.3151350, 0.3242053
1: -0.1052670, 0.1127120, -0.1168623, 0.1270175, -0.2322845, 0.2295743
2: -0.1008987, 0.1787496, -0.1095840, 0.1973391, -0.2982378, 0.2883336
3: 0.4196393, 1.0340916, 0.3535259, 1.0441525, -0.6245132, 0.6805657
4: -0.1251058, 0.1190936, -0.1376660, 0.1310732, -0.2561790, 0.2567596
5: -0.0586379, 0.4646062, -0.0681767, 0.5157401, -0.5743780, 0.5327829
6: -0.1318853, 0.1461769, -0.1447474, 0.1678536, -0.2997389, 0.2909243
7: -0.1864930, 0.1413886, -0.2137316, 0.1502388, -0.3367318, 0.3551203
8: -0.0932197, 0.1716688, -0.1186267, 0.1933837, -0.2866034, 0.2902955
9: -0.2185586, 0.1716712, -0.2485444, 0.1931556, -0.4117143, 0.4202156

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 2.13 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1207468, 0.1308932, -0.1559681, 0.1576106, -0.2783574, 0.2868614
1: -0.0882468, 0.0960897, -0.1077271, 0.1179539, -0.2062007, 0.2038168
2: -0.0899592, 0.1514581, -0.1031826, 0.1838737, -0.2738329, 0.2546407
3: 0.4851008, 1.0281571, 0.3858503, 1.0409738, -0.5558729, 0.6423069
4: -0.1048785, 0.1029995, -0.1265108, 0.1218139, -0.2266925, 0.2295103
5: -0.0474810, 0.4096504, -0.0600203, 0.4905151, -0.5379961, 0.4696707
6: -0.1147075, 0.1198550, -0.1347217, 0.1549277, -0.2696352, 0.2545767
7: -0.1586763, 0.1267411, -0.1996915, 0.1429581, -0.3016344, 0.3264326
8: -0.0671329, 0.1537934, -0.1051887, 0.1788850, -0.2460179, 0.2589822
9: -0.1936240, 0.1493764, -0.2339030, 0.1773981, -0.3710221, 0.3832794

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7977373, upper bound: 0.7929652
time: 1.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909359
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1176214, 0.1280110, -0.2226183, 0.2166913, -0.3343127, 0.3506293
1: -0.0862550, 0.0938514, -0.1505585, 0.1669702, -0.2532251, 0.2444099
2: -0.0884515, 0.1485549, -0.1358902, 0.2451377, -0.3335891, 0.2844451
3: 0.4953572, 1.0276734, 0.2258523, 1.0527583, -0.5574011, 0.8018211
4: -0.1033042, 0.1016172, -0.1794716, 0.1654891, -0.2687933, 0.2810888
5: -0.0466779, 0.4008894, -0.1025296, 0.6179978, -0.6646757, 0.5034190
6: -0.1130880, 0.1162653, -0.1822927, 0.2157518, -0.3288398, 0.2985580
7: -0.1542481, 0.1253109, -0.2694196, 0.1803909, -0.3346390, 0.3947305
8: -0.0635725, 0.1520059, -0.1688992, 0.2478859, -0.3114584, 0.3209051
9: -0.1903824, 0.1479826, -0.3064381, 0.2488165, -0.4391988, 0.4544207

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 144

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0960476, 0.1079945, -0.2081929, 0.2038512, -0.2998988, 0.3161874
1: -0.0727019, 0.0786828, -0.1405014, 0.1561741, -0.2288761, 0.2191842
2: -0.0781674, 0.1257948, -0.1277822, 0.2318663, -0.3100337, 0.2535770
3: 0.5512632, 1.0239917, 0.2588924, 1.0498757, -0.4986125, 0.7650993
4: -0.0873080, 0.0880874, -0.1672998, 0.1555107, -0.2428186, 0.2553872
5: -0.0397462, 0.3528910, -0.0918130, 0.5923262, -0.6320724, 0.4447040
6: -0.0979064, 0.0935319, -0.1721295, 0.2026924, -0.3005988, 0.2656614
7: -0.1321757, 0.1110208, -0.2547638, 0.1702741, -0.3024498, 0.3657845
8: -0.0453001, 0.1356346, -0.1552671, 0.2326665, -0.2779666, 0.2909017
9: -0.1687368, 0.1361394, -0.2912222, 0.2318400, -0.4005768, 0.4273616

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1177172, 0.1280896, -0.2506177, 0.2448889, -0.3626061, 0.3787073
1: -0.0863102, 0.0939018, -0.1741652, 0.1883706, -0.2746808, 0.2680671
2: -0.0884963, 0.1486507, -0.1558241, 0.2726204, -0.3611167, 0.3044748
3: 0.4951157, 1.0277025, 0.1661890, 1.0572103, -0.5620946, 0.8615135
4: -0.1033555, 0.1016615, -0.2056279, 0.1898767, -0.2932322, 0.3072894
5: -0.0467021, 0.4010797, -0.1263336, 0.6648675, -0.7115695, 0.5274133
6: -0.1131378, 0.1163656, -0.2029888, 0.2445784, -0.3577162, 0.3193544
7: -0.1543467, 0.1253784, -0.2987874, 0.2031641, -0.3575107, 0.4241658
8: -0.0636586, 0.1520712, -0.1974593, 0.2785859, -0.3422445, 0.3495305
9: -0.1904658, 0.1480583, -0.3342946, 0.2861739, -0.4766397, 0.4823530

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0961347, 0.1080595, -0.2356418, 0.2299536, -0.3260883, 0.3437013
1: -0.0727482, 0.0787322, -0.1627637, 0.1772753, -0.2500236, 0.2414960
2: -0.0782072, 0.1258768, -0.1461354, 0.2589366, -0.3371438, 0.2720121
3: 0.5510182, 1.0240104, 0.2011491, 1.0542927, -0.5032745, 0.8228613
4: -0.0873502, 0.0881315, -0.1930673, 0.1777994, -0.2651496, 0.2811988
5: -0.0397632, 0.3530774, -0.1150068, 0.6375265, -0.6772897, 0.4680842
6: -0.0979516, 0.0936279, -0.1924996, 0.2293131, -0.3272647, 0.2861275
7: -0.1322583, 0.1110893, -0.2823298, 0.1927167, -0.3249750, 0.3934191
8: -0.0453701, 0.1357013, -0.1816034, 0.2627561, -0.3081262, 0.3173048
9: -0.1688166, 0.1362091, -0.3184251, 0.2669476, -0.4357642, 0.4546342

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
time: 1.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.2028642, 0.2011256, -0.3465115, 0.3570303
1: -0.1052670, 0.1127120, -0.1432786, 0.1543447, -0.2596116, 0.2559906
2: -0.1008987, 0.1787496, -0.1300209, 0.2327995, -0.3336983, 0.3087705
3: 0.4196393, 1.0340916, 0.2828985, 1.0483516, -0.6287124, 0.7511931
4: -0.1251058, 0.1190936, -0.1714482, 0.1572951, -0.2824008, 0.2905417
5: -0.0586379, 0.4646062, -0.0961161, 0.5694689, -0.6281067, 0.5607222
6: -0.1318853, 0.1461769, -0.1714040, 0.1990226, -0.3309079, 0.3175809
7: -0.1864930, 0.1413886, -0.2463606, 0.1755461, -0.3620391, 0.3877493
8: -0.0932197, 0.1716688, -0.1487307, 0.2303757, -0.3235954, 0.3203995
9: -0.2185586, 0.1716712, -0.2799888, 0.2392697, -0.4578283, 0.4516600

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1207468, 0.1308932, -0.1870677, 0.1868715, -0.3076183, 0.3179609
1: -0.0882468, 0.0960897, -0.1323955, 0.1422783, -0.2305251, 0.2284852
2: -0.0899592, 0.1514581, -0.1214969, 0.2179812, -0.3079404, 0.2729550
3: 0.4851008, 1.0281571, 0.3200509, 1.0447118, -0.5596110, 0.7081063
4: -0.1048785, 0.1029995, -0.1583137, 0.1463817, -0.2512603, 0.2613132
5: -0.0474810, 0.4096504, -0.0848192, 0.5408301, -0.5883111, 0.4944696
6: -0.1147075, 0.1198550, -0.1601073, 0.1845536, -0.2992611, 0.2799623
7: -0.1586763, 0.1267411, -0.2298687, 0.1646079, -0.3232842, 0.3566098
8: -0.0671329, 0.1537934, -0.1336126, 0.2133581, -0.2804910, 0.2874060
9: -0.1936240, 0.1493764, -0.2628790, 0.2206163, -0.4142404, 0.4122554

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013759
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.1698477, 0.1738739, -0.3192598, 0.3240138
1: -0.1052670, 0.1127120, -0.1215079, 0.1292673, -0.2345343, 0.2342199
2: -0.1008987, 0.1787496, -0.1132401, 0.2025333, -0.3034321, 0.2919897
3: 0.4196393, 1.0340916, 0.3614141, 1.0416410, -0.6220017, 0.6726775
4: -0.1251058, 0.1190936, -0.1453356, 0.1350990, -0.2602048, 0.2644292
5: -0.0586379, 0.4646062, -0.0741304, 0.5064704, -0.5651082, 0.5387365
6: -0.1318853, 0.1461769, -0.1478697, 0.1689372, -0.3008225, 0.2940466
7: -0.1864930, 0.1413886, -0.2114801, 0.1550510, -0.3415440, 0.3528688
8: -0.0932197, 0.1716688, -0.1174472, 0.1957375, -0.2889572, 0.2891160
9: -0.2185586, 0.1716712, -0.2433822, 0.2037327, -0.4222913, 0.4150534

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1207468, 0.1308932, -0.1546687, 0.1615878, -0.2823346, 0.2855619
1: -0.0882468, 0.0960897, -0.1115133, 0.1177644, -0.2060112, 0.2076030
2: -0.0899592, 0.1514581, -0.1052911, 0.1879223, -0.2778815, 0.2567492
3: 0.4851008, 1.0281571, 0.3971850, 1.0379288, -0.5528280, 0.6309721
4: -0.1048785, 0.1029995, -0.1324655, 0.1249762, -0.2298547, 0.2354651
5: -0.0474810, 0.4096504, -0.0641854, 0.4792956, -0.5267766, 0.4738358
6: -0.1147075, 0.1198550, -0.1374239, 0.1550166, -0.2697241, 0.2572789
7: -0.1586763, 0.1267411, -0.1956195, 0.1471029, -0.3057792, 0.3223606
8: -0.0671329, 0.1537934, -0.1029415, 0.1791309, -0.2462638, 0.2567349
9: -0.1936240, 0.1493764, -0.2276118, 0.1849582, -0.3785823, 0.3769882

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7977373, upper bound: 0.7929652
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976767, upper bound: 0.7912324
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1183584, 0.1287085, -0.1823466, 0.1848758, -0.3032342, 0.3110551
1: -0.0867417, 0.0942739, -0.1308306, 0.1398219, -0.2265635, 0.2251045
2: -0.0888284, 0.1493551, -0.1201123, 0.2151396, -0.3039680, 0.2694674
3: 0.4937043, 1.0278168, 0.3331147, 1.0416272, -0.5479229, 0.6947020
4: -0.1038491, 0.1020425, -0.1565190, 0.1446831, -0.2485322, 0.2585615
5: -0.0469194, 0.4022181, -0.0833027, 0.5309573, -0.5778767, 0.4855208
6: -0.1135887, 0.1170263, -0.1577751, 0.1807420, -0.2943307, 0.2748014
7: -0.1549252, 0.1258640, -0.2243968, 0.1637465, -0.3186717, 0.3502608
8: -0.0641901, 0.1526139, -0.1283412, 0.2087233, -0.2729135, 0.2809552
9: -0.1910248, 0.1484303, -0.2566607, 0.2154681, -0.4064929, 0.4050910

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7911804, upper bound: 0.7987355
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7911438, upper bound: 0.7963659
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0967106, 0.1086234, -0.1823466, 0.1848758, -0.2815864, 0.2909700
1: -0.0731243, 0.0791069, -0.1308306, 0.1398219, -0.2129461, 0.2099375
2: -0.0785225, 0.1264241, -0.1201123, 0.2151396, -0.2936621, 0.2465364
3: 0.5495667, 1.0240868, 0.3331147, 1.0416272, -0.4920604, 0.6909721
4: -0.0877954, 0.0885188, -0.1565190, 0.1446831, -0.2324785, 0.2450378
5: -0.0399465, 0.3542351, -0.0833027, 0.5309573, -0.5709038, 0.4375378
6: -0.0983854, 0.0942772, -0.1577751, 0.1807420, -0.2791274, 0.2520524
7: -0.1327828, 0.1115826, -0.2243968, 0.1637465, -0.2965293, 0.3359795
8: -0.0458068, 0.1362510, -0.1283412, 0.2087233, -0.2545302, 0.2645922
9: -0.1693630, 0.1365667, -0.2566607, 0.2154681, -0.3848311, 0.3932275

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7911804, upper bound: 0.7987355
time: 1.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7911438, upper bound: 0.7963659
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1433526, 0.1515372, -0.1527154, 0.1607351, -0.3040876, 0.3042527
1: -0.1031639, 0.1114677, -0.1104556, 0.1163444, -0.2195083, 0.2219232
2: -0.0997586, 0.1759580, -0.1043601, 0.1861702, -0.2859288, 0.2803181
3: 0.4196135, 1.0354754, 0.4023970, 1.0352353, -0.6156218, 0.6330783
4: -0.1225686, 0.1169013, -0.1303014, 0.1236413, -0.2462099, 0.2472026
5: -0.0571293, 0.4626505, -0.0619224, 0.4771147, -0.5342439, 0.5245729
6: -0.1294274, 0.1444206, -0.1365613, 0.1535418, -0.2829692, 0.2809819
7: -0.1855835, 0.1398271, -0.1932238, 0.1462515, -0.3318351, 0.3330509
8: -0.0923306, 0.1687555, -0.1000599, 0.1777478, -0.2700784, 0.2688154
9: -0.2173271, 0.1707974, -0.2253275, 0.1782827, -0.3956097, 0.3961249

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7926662, upper bound: 0.7910332
time: 1.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7926662, upper bound: 0.7910332
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1900263, 0.1910230, -0.1528455, 0.1608297, -0.3508559, 0.3438685
1: -0.1352142, 0.1451717, -0.1105327, 0.1163962, -0.2516104, 0.2557045
2: -0.1239960, 0.2224609, -0.1044118, 0.1862868, -0.3102828, 0.3268727
3: 0.3120077, 1.0441910, 0.4021413, 1.0352788, -0.7232711, 0.6420497
4: -0.1628236, 0.1489002, -0.1303765, 0.1237125, -0.2865361, 0.2792767
5: -0.0872374, 0.5439036, -0.0619751, 0.4773046, -0.5645420, 0.6058788
6: -0.1632349, 0.1879889, -0.1366250, 0.1536577, -0.3168927, 0.3246139
7: -0.2323336, 0.1661087, -0.1933362, 0.1463276, -0.3786611, 0.3594449
8: -0.1356682, 0.2179951, -0.1001812, 0.1778360, -0.3135042, 0.3181763
9: -0.2641295, 0.2265409, -0.2254584, 0.1784256, -0.4425550, 0.4519993

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909993, upper bound: 0.7910107
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909993, upper bound: 0.7910107
time: 1.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.37 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013185
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988673
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7977373, upper bound: 0.7929228
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909334
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013185
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7988727, upper bound: 0.7988673
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8005959, upper bound: 0.7931865
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7979024, upper bound: 0.7929228
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7978998, upper bound: 0.7912154
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7908841, upper bound: 0.7982595
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7908841, upper bound: 0.7982595
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013759
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7977373, upper bound: 0.7929652
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909359
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8016928, upper bound: 0.8016928
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7989205, upper bound: 0.8013759
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.8004487, upper bound: 0.7931865
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7977373, upper bound: 0.7929652
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7976767, upper bound: 0.7912324
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7911804, upper bound: 0.7987355
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7911438, upper bound: 0.7963659
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7911804, upper bound: 0.7987355
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7911438, upper bound: 0.7963659
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7926662, upper bound: 0.7910332
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7926662, upper bound: 0.7910332
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7909993, upper bound: 0.7910107
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -0.7909993, upper bound: 0.7910107

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1323535, 0.1398139, -0.2721674, 0.2721674
1: -0.0939614, 0.1055578, -0.0939614, 0.1055578, -0.1995192, 0.1995192
2: -0.0941783, 0.1624106, -0.0941783, 0.1624106, -0.2565889, 0.2565889
3: 0.4443746, 1.0336604, 0.4443746, 1.0336604, -0.5892859, 0.5892859
4: -0.1111998, 0.1083961, -0.1111998, 0.1083961, -0.2195958, 0.2195958
5: -0.0505697, 0.4460281, -0.0505697, 0.4460281, -0.4965978, 0.4965978
6: -0.1205043, 0.1328021, -0.1205043, 0.1328021, -0.2533064, 0.2533064
7: -0.1758752, 0.1307725, -0.1758752, 0.1307725, -0.3066478, 0.3066478
8: -0.0818439, 0.1581148, -0.0818439, 0.1581148, -0.2399587, 0.2399587
9: -0.2079326, 0.1538350, -0.2079326, 0.1538350, -0.3617676, 0.3617676

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8070467, upper bound: 0.7990571
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8043608, upper bound: 0.7990096
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1186917, 0.1269170, -0.2592705, 0.2585056
1: -0.0939614, 0.1055578, -0.0852304, 0.0961525, -0.1901139, 0.1907883
2: -0.0941783, 0.1624106, -0.0878181, 0.1474300, -0.2416083, 0.2502286
3: 0.4443746, 1.0336604, 0.4772729, 1.0306065, -0.5862319, 0.5563875
4: -0.1111998, 0.1083961, -0.1003226, 0.0999454, -0.2111451, 0.2087186
5: -0.0505697, 0.4460281, -0.0457663, 0.4164358, -0.4670055, 0.4917944
6: -0.1205043, 0.1328021, -0.1106552, 0.1185563, -0.2390606, 0.2434573
7: -0.1758752, 0.1307725, -0.1618960, 0.1219839, -0.2978591, 0.2926686
8: -0.0818439, 0.1581148, -0.0691035, 0.1477058, -0.2295497, 0.2272183
9: -0.2079326, 0.1538350, -0.1941239, 0.1472560, -0.3551886, 0.3479589

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8070467, upper bound: 0.7990571
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8043608, upper bound: 0.7990096
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1179813, 0.1262133, -0.1676116, 0.1666437, -0.2846250, 0.2938249
1: -0.0847356, 0.0957536, -0.1136909, 0.1257174, -0.2104530, 0.2094445
2: -0.0874326, 0.1466492, -0.1066725, 0.1935570, -0.2809895, 0.2533217
3: 0.4787382, 1.0304592, 0.3556925, 1.0427483, -0.5640101, 0.6747667
4: -0.0997479, 0.0995103, -0.1337138, 0.1287253, -0.2284732, 0.2332241
5: -0.0455168, 0.4152493, -0.0649723, 0.5173870, -0.5629038, 0.4802216
6: -0.1101324, 0.1178259, -0.1424800, 0.1651065, -0.2752388, 0.2603059
7: -0.1613001, 0.1214062, -0.2127165, 0.1462371, -0.3075372, 0.3341227
8: -0.0685124, 0.1470593, -0.1168307, 0.1890369, -0.2575494, 0.2638900
9: -0.1934909, 0.1468323, -0.2480972, 0.1849798, -0.3784707, 0.3949295

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988673
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988673
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1180241, 0.1262413, -0.1963227, 0.1938635, -0.3118876, 0.3225640
1: -0.0847567, 0.0957679, -0.1339435, 0.1477178, -0.2324745, 0.2297114
2: -0.0874521, 0.1466881, -0.1227878, 0.2224910, -0.3099431, 0.2694759
3: 0.4786510, 1.0304844, 0.2912362, 1.0473576, -0.5687066, 0.7392483
4: -0.0997651, 0.0995247, -0.1602450, 0.1484990, -0.2482641, 0.2597697
5: -0.0455256, 0.4153125, -0.0850894, 0.5667512, -0.6122769, 0.5004019
6: -0.1101478, 0.1178702, -0.1646160, 0.1921706, -0.3023184, 0.2824862
7: -0.1613395, 0.1214368, -0.2415616, 0.1629285, -0.3242679, 0.3629984
8: -0.0685580, 0.1470857, -0.1436460, 0.2210789, -0.2896369, 0.2907317
9: -0.1935293, 0.1468638, -0.2765329, 0.2212478, -0.4147772, 0.4233966

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988673
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988673
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1131819, 0.1219674, -0.2543209, 0.2529958
1: -0.0939614, 0.1055578, -0.0819143, 0.0920996, -0.1860610, 0.1874721
2: -0.0941783, 0.1624106, -0.0852243, 0.1422078, -0.2363862, 0.2476349
3: 0.4443746, 1.0336604, 0.4948269, 1.0293642, -0.5849897, 0.5388335
4: -0.1111998, 0.1083961, -0.0972459, 0.0972678, -0.2084676, 0.2056420
5: -0.0505697, 0.4460281, -0.0444142, 0.4013967, -0.4519664, 0.4904423
6: -0.1205043, 0.1328021, -0.1075819, 0.1124838, -0.2329881, 0.2403840
7: -0.1758752, 0.1307725, -0.1546251, 0.1193287, -0.2952040, 0.2853977
8: -0.0818439, 0.1581148, -0.0627786, 0.1446541, -0.2264980, 0.2208934
9: -0.2079326, 0.1538350, -0.1881767, 0.1446504, -0.3525830, 0.3420117

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8040244, upper bound: 0.7911140
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8021425, upper bound: 0.7910631
time: 1.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1008312, 0.1093591, -0.2417126, 0.2406451
1: -0.0939614, 0.1055578, -0.0737615, 0.0830829, -0.1770444, 0.1793193
2: -0.0941783, 0.1624106, -0.0790291, 0.1297382, -0.2239165, 0.2414396
3: 0.4443746, 1.0336604, 0.5273200, 1.0266857, -0.5823112, 0.5063404
4: -0.1111998, 0.1083961, -0.0868989, 0.0889649, -0.2001647, 0.1952950
5: -0.0505697, 0.4460281, -0.0397120, 0.3735656, -0.4241353, 0.4857401
6: -0.1205043, 0.1328021, -0.0980690, 0.0989287, -0.2194330, 0.2308710
7: -0.1758752, 0.1307725, -0.1409962, 0.1107113, -0.2865865, 0.2717687
8: -0.0818439, 0.1581148, -0.0514807, 0.1345519, -0.2163958, 0.2095955
9: -0.2079326, 0.1538350, -0.1750948, 0.1379746, -0.3459072, 0.3289298

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8040244, upper bound: 0.7911140
time: 2.45 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8021425, upper bound: 0.7910631
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1179813, 0.1262133, -0.1448753, 0.1472884, -0.2652697, 0.2710886
1: -0.0847356, 0.0957536, -0.0990862, 0.1126179, -0.1973535, 0.1948398
2: -0.0874326, 0.1466492, -0.0980847, 0.1718817, -0.2593142, 0.2447339
3: 0.4787382, 1.0304592, 0.4069002, 1.0387719, -0.5600337, 0.6235589
4: -0.0997479, 0.0995103, -0.1165836, 0.1136026, -0.2133506, 0.2160939
5: -0.0455168, 0.4152493, -0.0532438, 0.4745703, -0.5200871, 0.4684930
6: -0.1101324, 0.1178259, -0.1264503, 0.1444066, -0.2545390, 0.2442762
7: -0.1613001, 0.1214062, -0.1904205, 0.1353994, -0.2966995, 0.3118267
8: -0.0685124, 0.1470593, -0.0949170, 0.1676144, -0.2361268, 0.2419763
9: -0.1934909, 0.1468323, -0.2240686, 0.1639596, -0.3574505, 0.3709009

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909334
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909334
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1180241, 0.1262413, -0.1683605, 0.1682937, -0.2863178, 0.2946018
1: -0.0847567, 0.0957679, -0.1163364, 0.1248079, -0.2095646, 0.2121043
2: -0.0874521, 0.1466881, -0.1081156, 0.1964567, -0.2839088, 0.2548037
3: 0.4786510, 1.0304844, 0.3592461, 1.0426173, -0.5639664, 0.6712383
4: -0.0997651, 0.0995247, -0.1365195, 0.1304664, -0.2302315, 0.2360442
5: -0.0455256, 0.4153125, -0.0667525, 0.5125632, -0.5580888, 0.4820650
6: -0.1101478, 0.1178702, -0.1437285, 0.1665697, -0.2767175, 0.2615986
7: -0.1613395, 0.1214368, -0.2112145, 0.1497268, -0.3110663, 0.3326513
8: -0.0685580, 0.1470857, -0.1165961, 0.1910026, -0.2595606, 0.2636818
9: -0.1935293, 0.1468638, -0.2464552, 0.1890608, -0.3825901, 0.3933190

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909334
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909334
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1124734, 0.1212652, -0.1606673, 0.1615451, -0.2740186, 0.2819326
1: -0.0814218, 0.0917020, -0.1105849, 0.1220409, -0.2034626, 0.2022869
2: -0.0848431, 0.1414255, -0.1044699, 0.1884535, -0.2732965, 0.2458954
3: 0.4962844, 1.0292321, 0.3743249, 1.0392920, -0.5430076, 0.6549072
4: -0.0966727, 0.0968332, -0.1302291, 0.1253488, -0.2220215, 0.2270623
5: -0.0441654, 0.4002295, -0.0622286, 0.5033692, -0.5475346, 0.4624582
6: -0.1070608, 0.1117572, -0.1385837, 0.1592154, -0.2662762, 0.2503409
7: -0.1540299, 0.1187557, -0.2049520, 0.1439491, -0.2979790, 0.3237076
8: -0.0621860, 0.1440119, -0.1091712, 0.1822783, -0.2444642, 0.2531831
9: -0.1875473, 0.1442345, -0.2393888, 0.1774941, -0.3650414, 0.3836233

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928173
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928173
time: 2.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1124734, 0.1212652, -0.1442841, 0.1478183, -0.2602917, 0.2655493
1: -0.0814218, 0.0917020, -0.0994391, 0.1127605, -0.1941823, 0.1911411
2: -0.0848431, 0.1414255, -0.0983008, 0.1717905, -0.2566336, 0.2397264
3: 0.4962844, 1.0292321, 0.4094378, 1.0362738, -0.5399894, 0.6197944
4: -0.0966727, 0.0968332, -0.1164288, 0.1137928, -0.2104655, 0.2132620
5: -0.0441654, 0.4002295, -0.0530722, 0.4738035, -0.5179689, 0.4533017
6: -0.1070608, 0.1117572, -0.1267351, 0.1440900, -0.2511508, 0.2384923
7: -0.1540299, 0.1187557, -0.1893097, 0.1354625, -0.2894923, 0.3080653
8: -0.0621860, 0.1440119, -0.0933236, 0.1670191, -0.2292050, 0.2373355
9: -0.1875473, 0.1442345, -0.2224768, 0.1607780, -0.3483254, 0.3667113

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928173
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928173
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1002065, 0.1086552, -0.1676116, 0.1666437, -0.2668502, 0.2762668
1: -0.0733034, 0.0826874, -0.1136909, 0.1257174, -0.1990208, 0.1963783
2: -0.0786476, 0.1291508, -0.1066725, 0.1935570, -0.2722046, 0.2358234
3: 0.5287691, 1.0265702, 0.3556925, 1.0427483, -0.5139792, 0.6708777
4: -0.0863403, 0.0885295, -0.1337138, 0.1287253, -0.2150656, 0.2222433
5: -0.0394679, 0.3723960, -0.0649723, 0.5173870, -0.5568549, 0.4373683
6: -0.0975457, 0.0982279, -0.1424800, 0.1651065, -0.2626522, 0.2407079
7: -0.1404083, 0.1101410, -0.2127165, 0.1462371, -0.2866454, 0.3228574
8: -0.0509891, 0.1339294, -0.1168307, 0.1890369, -0.2400260, 0.2507601
9: -0.1745202, 0.1375674, -0.2480972, 0.1849798, -0.3595000, 0.3856646

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1002065, 0.1086552, -0.1448753, 0.1472884, -0.2474949, 0.2535305
1: -0.0733034, 0.0826874, -0.0990862, 0.1126179, -0.1859213, 0.1817736
2: -0.0786476, 0.1291508, -0.0980847, 0.1718817, -0.2505293, 0.2272356
3: 0.5287691, 1.0265702, 0.4069002, 1.0387719, -0.5100027, 0.6196700
4: -0.0863403, 0.0885295, -0.1165836, 0.1136026, -0.1999429, 0.2051131
5: -0.0394679, 0.3723960, -0.0532438, 0.4745703, -0.5140381, 0.4256397
6: -0.0975457, 0.0982279, -0.1264503, 0.1444066, -0.2419523, 0.2246782
7: -0.1404083, 0.1101410, -0.1904205, 0.1353994, -0.2758077, 0.3005614
8: -0.0509891, 0.1339294, -0.0949170, 0.1676144, -0.2186035, 0.2288465
9: -0.1745202, 0.1375674, -0.2240686, 0.1639596, -0.3384799, 0.3616360

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907381, upper bound: 0.7926658
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1124999, 0.1212799, -0.1896216, 0.1880180, -0.3005179, 0.3109015
1: -0.0814334, 0.0917082, -0.1318810, 0.1436484, -0.2250818, 0.2235891
2: -0.0848542, 0.1414485, -0.1208613, 0.2178894, -0.3027437, 0.2623098
3: 0.4962292, 1.0292531, 0.3108215, 1.0440311, -0.5478020, 0.7184316
4: -0.0966792, 0.0968388, -0.1572265, 0.1460620, -0.2427412, 0.2540653
5: -0.0441699, 0.4002717, -0.0834965, 0.5518676, -0.5960374, 0.4837682
6: -0.1070660, 0.1117856, -0.1608689, 0.1865107, -0.2935767, 0.2726545
7: -0.1540556, 0.1187765, -0.2335046, 0.1630585, -0.3171142, 0.3522810
8: -0.0622183, 0.1440273, -0.1364335, 0.2143974, -0.2766157, 0.2804608
9: -0.1875719, 0.1442588, -0.2678609, 0.2140354, -0.4016073, 0.4121197

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908337
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908337
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1124999, 0.1212799, -0.1692388, 0.1696041, -0.2821040, 0.2905188
1: -0.0814334, 0.0917082, -0.1171361, 0.1269905, -0.2084239, 0.2088443
2: -0.0848542, 0.1414485, -0.1095088, 0.1972732, -0.2821275, 0.2509573
3: 0.4962292, 1.0292531, 0.3563203, 1.0405133, -0.5442841, 0.6729328
4: -0.0966792, 0.0968388, -0.1374932, 0.1310381, -0.2277173, 0.2343320
5: -0.0441699, 0.4002717, -0.0672841, 0.5154582, -0.5596281, 0.4675557
6: -0.1070660, 0.1117856, -0.1449645, 0.1676442, -0.2747102, 0.2567502
7: -0.1540556, 0.1187765, -0.2123482, 0.1502874, -0.3043430, 0.3311246
8: -0.0622183, 0.1440273, -0.1165655, 0.1923174, -0.2545357, 0.2605928
9: -0.1875719, 0.1442588, -0.2467918, 0.1877160, -0.3752880, 0.3910506

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908337
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908337
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1002313, 0.1086699, -0.1963874, 0.1938635, -0.2940948, 0.3050573
1: -0.0733144, 0.0826953, -0.1345346, 0.1477779, -0.2210923, 0.2172299
2: -0.0786585, 0.1291733, -0.1231195, 0.2225223, -0.3011808, 0.2522928
3: 0.5287286, 1.0265849, 0.2911931, 1.0473754, -0.5186468, 0.7353917
4: -0.0863500, 0.0885365, -0.1602945, 0.1489823, -0.2353323, 0.2488310
5: -0.0394732, 0.3724322, -0.0860005, 0.5667777, -0.6062509, 0.4584327
6: -0.0975534, 0.0982533, -0.1646202, 0.1922804, -0.2898339, 0.2628734
7: -0.1404310, 0.1101577, -0.2415916, 0.1649047, -0.3053357, 0.3517492
8: -0.0510122, 0.1339464, -0.1436770, 0.2211606, -0.2721728, 0.2776234
9: -0.1745385, 0.1375798, -0.2768697, 0.2212478, -0.3957863, 0.4144494

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1002313, 0.1086699, -0.1683605, 0.1683020, -0.2685333, 0.2770304
1: -0.0733144, 0.0826953, -0.1163364, 0.1262599, -0.1995744, 0.1990317
2: -0.0786585, 0.1291733, -0.1090095, 0.1964567, -0.2751152, 0.2381828
3: 0.5287286, 1.0265849, 0.3585517, 1.0426173, -0.5138887, 0.6680331
4: -0.0863500, 0.0885365, -0.1372469, 0.1304664, -0.2168164, 0.2257833
5: -0.0394732, 0.3724322, -0.0677316, 0.5125632, -0.5520364, 0.4401638
6: -0.0975534, 0.0982533, -0.1440229, 0.1665697, -0.2641231, 0.2422762
7: -0.1404310, 0.1101577, -0.2117762, 0.1497268, -0.2901579, 0.3219339
8: -0.0510122, 0.1339464, -0.1165961, 0.1918102, -0.2428224, 0.2505424
9: -0.1745385, 0.1375798, -0.2464552, 0.1903921, -0.3649306, 0.3840350

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7906845, upper bound: 0.7906845
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1453859, 0.1541661, -0.2865196, 0.2851998
1: -0.0939614, 0.1055578, -0.1052670, 0.1127120, -0.2066734, 0.2108248
2: -0.0941783, 0.1624106, -0.1008987, 0.1787496, -0.2729279, 0.2633093
3: 0.4443746, 1.0336604, 0.4196393, 1.0340916, -0.5897170, 0.6140211
4: -0.1111998, 0.1083961, -0.1251058, 0.1190936, -0.2302933, 0.2335018
5: -0.0505697, 0.4460281, -0.0586379, 0.4646062, -0.5151759, 0.5046660
6: -0.1205043, 0.1328021, -0.1318853, 0.1461769, -0.2666812, 0.2646874
7: -0.1758752, 0.1307725, -0.1864930, 0.1413886, -0.3172639, 0.3172656
8: -0.0818439, 0.1581148, -0.0932197, 0.1716688, -0.2535127, 0.2513345
9: -0.2079326, 0.1538350, -0.2185586, 0.1716712, -0.3796038, 0.3723936

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8073392, upper bound: 0.7990571
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8044079, upper bound: 0.7990096
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1207468, 0.1308932, -0.2632467, 0.2605607
1: -0.0939614, 0.1055578, -0.0882468, 0.0960897, -0.1900511, 0.1938046
2: -0.0941783, 0.1624106, -0.0899592, 0.1514581, -0.2456364, 0.2523698
3: 0.4443746, 1.0336604, 0.4851008, 1.0281571, -0.5837826, 0.5485595
4: -0.1111998, 0.1083961, -0.1048785, 0.1029995, -0.2141993, 0.2132746
5: -0.0505697, 0.4460281, -0.0474810, 0.4096504, -0.4602201, 0.4935091
6: -0.1205043, 0.1328021, -0.1147075, 0.1198550, -0.2403593, 0.2475096
7: -0.1758752, 0.1307725, -0.1586763, 0.1267411, -0.3026164, 0.2894488
8: -0.0818439, 0.1581148, -0.0671329, 0.1537934, -0.2356374, 0.2252477
9: -0.2079326, 0.1538350, -0.1936240, 0.1493764, -0.3573090, 0.3474591

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8073392, upper bound: 0.7990571
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8044079, upper bound: 0.7990096
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1179813, 0.1262133, -0.1756480, 0.1768111, -0.2947924, 0.3018613
1: -0.0847356, 0.0957536, -0.1234002, 0.1339421, -0.2186777, 0.2191538
2: -0.0874326, 0.1466492, -0.1143684, 0.2060339, -0.2934664, 0.2610176
3: 0.4787382, 1.0304592, 0.3432956, 1.0422076, -0.5634694, 0.6871636
4: -0.0997479, 0.0995103, -0.1474560, 0.1377347, -0.2374826, 0.2469663
5: -0.0455168, 0.4152493, -0.0754830, 0.5242273, -0.5697441, 0.4907323
6: -0.1101324, 0.1178259, -0.1513449, 0.1737776, -0.2839100, 0.2691707
7: -0.1613001, 0.1214062, -0.2190597, 0.1551920, -0.3164921, 0.3404660
8: -0.0685124, 0.1470593, -0.1230814, 0.2000790, -0.2685914, 0.2701407
9: -0.1934909, 0.1468323, -0.2520761, 0.2044821, -0.3979729, 0.3989084

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988727, upper bound: 0.7988673
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988727, upper bound: 0.7988673
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1180241, 0.1262413, -0.2269478, 0.2230245, -0.3410487, 0.3531891
1: -0.0847567, 0.0957679, -0.1604664, 0.1735150, -0.2582718, 0.2562343
2: -0.0874521, 0.1466881, -0.1431072, 0.2558318, -0.3432840, 0.2897953
3: 0.4786510, 1.0304844, 0.2245176, 1.0514159, -0.5727650, 0.8059668
4: -0.0997651, 0.0995247, -0.1920624, 0.1740631, -0.2738281, 0.2915871
5: -0.0455256, 0.4153125, -0.1134614, 0.6146414, -0.6601670, 0.5287739
6: -0.1101478, 0.1178702, -0.1891058, 0.2215713, -0.3317191, 0.3069760
7: -0.1613395, 0.1214368, -0.2713372, 0.1935918, -0.3549312, 0.3927740
8: -0.0685580, 0.1470857, -0.1713201, 0.2562427, -0.3248007, 0.3184059
9: -0.1935293, 0.1468638, -0.3056808, 0.2656768, -0.4592061, 0.4525446

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988727, upper bound: 0.7988673
time: 1.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988727, upper bound: 0.7988673
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.1183584, 0.1287085, -0.2610620, 0.2581723
1: -0.0939614, 0.1055578, -0.0867417, 0.0942739, -0.1882353, 0.1922995
2: -0.0941783, 0.1624106, -0.0888284, 0.1493551, -0.2435334, 0.2512389
3: 0.4443746, 1.0336604, 0.4937043, 1.0278168, -0.5834422, 0.5399562
4: -0.1111998, 0.1083961, -0.1038491, 0.1020425, -0.2132422, 0.2122452
5: -0.0505697, 0.4460281, -0.0469194, 0.4022181, -0.4527878, 0.4929475
6: -0.1205043, 0.1328021, -0.1135887, 0.1170263, -0.2375306, 0.2463908
7: -0.1758752, 0.1307725, -0.1549252, 0.1258640, -0.3017392, 0.2856978
8: -0.0818439, 0.1581148, -0.0641901, 0.1526139, -0.2344579, 0.2223049
9: -0.2079326, 0.1538350, -0.1910248, 0.1484303, -0.3563629, 0.3448598

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8056193, upper bound: 0.7914268
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8028406, upper bound: 0.7913628
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1323535, 0.1398139, -0.0967106, 0.1086234, -0.2409769, 0.2365245
1: -0.0939614, 0.1055578, -0.0731243, 0.0791069, -0.1730683, 0.1786821
2: -0.0941783, 0.1624106, -0.0785225, 0.1264241, -0.2206025, 0.2409330
3: 0.4443746, 1.0336604, 0.5495667, 1.0240868, -0.5797123, 0.4840937
4: -0.1111998, 0.1083961, -0.0877954, 0.0885188, -0.1997185, 0.1961915
5: -0.0505697, 0.4460281, -0.0399465, 0.3542351, -0.4048049, 0.4859746
6: -0.1205043, 0.1328021, -0.0983854, 0.0942772, -0.2147815, 0.2311875
7: -0.1758752, 0.1307725, -0.1327828, 0.1115826, -0.2874579, 0.2635553
8: -0.0818439, 0.1581148, -0.0458068, 0.1362510, -0.2180949, 0.2039216
9: -0.2079326, 0.1538350, -0.1693630, 0.1365667, -0.3444993, 0.3231980

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8056193, upper bound: 0.7914268
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8028406, upper bound: 0.7913628
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1179813, 0.1262133, -0.1433526, 0.1515372, -0.2695185, 0.2695659
1: -0.0847356, 0.0957536, -0.1031639, 0.1114677, -0.1962032, 0.1989176
2: -0.0874326, 0.1466492, -0.0997586, 0.1759580, -0.2633906, 0.2464077
3: 0.4787382, 1.0304592, 0.4196135, 1.0354754, -0.5567372, 0.6108457
4: -0.0997479, 0.0995103, -0.1225686, 0.1169013, -0.2166492, 0.2220789
5: -0.0455168, 0.4152493, -0.0571293, 0.4626505, -0.5081673, 0.4723786
6: -0.1101324, 0.1178259, -0.1294274, 0.1444206, -0.2545530, 0.2472533
7: -0.1613001, 0.1214062, -0.1855835, 0.1398271, -0.3011272, 0.3069898
8: -0.0685124, 0.1470593, -0.0923306, 0.1687555, -0.2372679, 0.2393899
9: -0.1934909, 0.1468323, -0.2173271, 0.1707974, -0.3642883, 0.3641593

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978998, upper bound: 0.7912154
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978998, upper bound: 0.7912154
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1180241, 0.1262413, -0.1900263, 0.1910230, -0.3090471, 0.3162675
1: -0.0847567, 0.0957679, -0.1352142, 0.1451717, -0.2299285, 0.2309821
2: -0.0874521, 0.1466881, -0.1239960, 0.2224609, -0.3099130, 0.2706841
3: 0.4786510, 1.0304844, 0.3120077, 1.0441910, -0.5655401, 0.7184768
4: -0.0997651, 0.0995247, -0.1628236, 0.1489002, -0.2486653, 0.2623484
5: -0.0455256, 0.4153125, -0.0872374, 0.5439036, -0.5894293, 0.5025499
6: -0.1101478, 0.1178702, -0.1632349, 0.1879889, -0.2981367, 0.2811051
7: -0.1613395, 0.1214368, -0.2323336, 0.1661087, -0.3274482, 0.3537704
8: -0.0685580, 0.1470857, -0.1356682, 0.2179951, -0.2865531, 0.2827539
9: -0.1935293, 0.1468638, -0.2641295, 0.2265409, -0.4200702, 0.4109932

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978998, upper bound: 0.7912154
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7978998, upper bound: 0.7912154
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1124734, 0.1212652, -0.1712665, 0.1750037, -0.2874771, 0.2925317
1: -0.0814218, 0.0917020, -0.1218794, 0.1316208, -0.2130425, 0.2135814
2: -0.0848431, 0.1414255, -0.1130464, 0.2033909, -0.2882340, 0.2544719
3: 0.4962844, 1.0292321, 0.3554932, 1.0392244, -0.5429400, 0.6737390
4: -0.0966727, 0.0968332, -0.1457617, 0.1360737, -0.2327464, 0.2425948
5: -0.0441654, 0.4002295, -0.0739254, 0.5150852, -0.5592506, 0.4741549
6: -0.1070608, 0.1117572, -0.1492110, 0.1702687, -0.2773295, 0.2609682
7: -0.1540299, 0.1187557, -0.2139341, 0.1540984, -0.3081283, 0.3326898
8: -0.0621860, 0.1440119, -0.1180838, 0.1958092, -0.2579952, 0.2620957
9: -0.1875473, 0.1442345, -0.2462257, 0.1996197, -0.3871671, 0.3904601

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7973338, upper bound: 0.7973311
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7973338, upper bound: 0.7973311
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1124999, 0.1212799, -0.2221429, 0.2188917, -0.3313916, 0.3434228
1: -0.0814334, 0.0917082, -0.1583742, 0.1708134, -0.2522468, 0.2500824
2: -0.0848542, 0.1414485, -0.1413152, 0.2524865, -0.3373407, 0.2827637
3: 0.4962292, 1.0292531, 0.2374596, 1.0486498, -0.5524206, 0.7917935
4: -0.0966792, 0.0968388, -0.1894254, 0.1719345, -0.2686137, 0.2862642
5: -0.0441699, 0.4002717, -0.1111712, 0.6050269, -0.6491967, 0.5114429
6: -0.1070660, 0.1117856, -0.1864354, 0.2175024, -0.3245685, 0.2982210
7: -0.1540556, 0.1187765, -0.2658400, 0.1920100, -0.3460656, 0.3846164
8: -0.0622183, 0.1440273, -0.1659950, 0.2512745, -0.3134928, 0.3100224
9: -0.1875719, 0.1442588, -0.2995841, 0.2595955, -0.4471675, 0.4438429

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7973338, upper bound: 0.7973311
time: 1.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7973338, upper bound: 0.7973311
time: 1.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1002065, 0.1086552, -0.1712665, 0.1750037, -0.2752102, 0.2799217
1: -0.0733034, 0.0826874, -0.1218794, 0.1316208, -0.2049242, 0.2045667
2: -0.0786476, 0.1291508, -0.1130464, 0.2033909, -0.2820385, 0.2421972
3: 0.5287691, 1.0265702, 0.3554932, 1.0392244, -0.5104553, 0.6710770
4: -0.0863403, 0.0885295, -0.1457617, 0.1360737, -0.2224140, 0.2342912
5: -0.0394679, 0.3723960, -0.0739254, 0.5150852, -0.5545530, 0.4463213
6: -0.0975457, 0.0982279, -0.1492110, 0.1702687, -0.2678144, 0.2474389
7: -0.1404083, 0.1101410, -0.2139341, 0.1540984, -0.2945067, 0.3240751
8: -0.0509891, 0.1339294, -0.1180838, 0.1958092, -0.2467983, 0.2520133
9: -0.1745202, 0.1375674, -0.2462257, 0.1996197, -0.3741400, 0.3837931

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
time: 1.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1002313, 0.1086699, -0.2221429, 0.2188917, -0.3191230, 0.3308128
1: -0.0733144, 0.0826953, -0.1583742, 0.1708134, -0.2441278, 0.2410695
2: -0.0786585, 0.1291733, -0.1413152, 0.2524865, -0.3311450, 0.2704886
3: 0.5287286, 1.0265849, 0.2374596, 1.0486498, -0.5199212, 0.7891252
4: -0.0863500, 0.0885365, -0.1894254, 0.1719345, -0.2582844, 0.2779619
5: -0.0394732, 0.3724322, -0.1111712, 0.6050269, -0.6445001, 0.4836034
6: -0.0975534, 0.0982533, -0.1864354, 0.2175024, -0.3150559, 0.2846887
7: -0.1404310, 0.1101577, -0.2658400, 0.1920100, -0.3324410, 0.3759976
8: -0.0510122, 0.1339464, -0.1659950, 0.2512745, -0.3022867, 0.2999414
9: -0.1745385, 0.1375798, -0.2995841, 0.2595955, -0.4341340, 0.4371638

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
time: 1.48 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7908369, upper bound: 0.7962326
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1448753, 0.1472884, -0.1200036, 0.1301872, -0.2750626, 0.2672919
1: -0.0990862, 0.1126179, -0.0877545, 0.0956622, -0.1947484, 0.2003725
2: -0.0980847, 0.1718817, -0.0895770, 0.1506519, -0.2487366, 0.2614587
3: 0.4069002, 1.0387719, 0.4867636, 1.0279950, -0.6210948, 0.5520083
4: -0.1165836, 0.1136026, -0.1043288, 0.1025711, -0.2191547, 0.2179314
5: -0.0532438, 0.4745703, -0.0472382, 0.4083028, -0.4615465, 0.5218084
6: -0.1264503, 0.1444066, -0.1142021, 0.1190824, -0.2455327, 0.2586086
7: -0.1904205, 0.1353994, -0.1579918, 0.1261811, -0.3166015, 0.2933912
8: -0.0949170, 0.1676144, -0.0664715, 0.1531795, -0.2480965, 0.2340859
9: -0.2240686, 0.1639596, -0.1929342, 0.1489232, -0.3729918, 0.3568938

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
time: 1.47 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1448753, 0.1472884, -0.0960476, 0.1079945, -0.2528698, 0.2433360
1: -0.0990862, 0.1126179, -0.0727019, 0.0786828, -0.1777690, 0.1853199
2: -0.0980847, 0.1718817, -0.0781674, 0.1257948, -0.2238795, 0.2500491
3: 0.4069002, 1.0387719, 0.5512632, 1.0239917, -0.6170915, 0.4875087
4: -0.1165836, 0.1136026, -0.0873080, 0.0880874, -0.2046710, 0.2009106
5: -0.0532438, 0.4745703, -0.0397462, 0.3528910, -0.4061348, 0.5143165
6: -0.1264503, 0.1444066, -0.0979064, 0.0935319, -0.2199821, 0.2423130
7: -0.1904205, 0.1353994, -0.1321757, 0.1110208, -0.3014412, 0.2675751
8: -0.0949170, 0.1676144, -0.0453001, 0.1356346, -0.2305516, 0.2129145
9: -0.2240686, 0.1639596, -0.1687368, 0.1361394, -0.3602080, 0.3326964

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
time: 1.27 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7927558, upper bound: 0.7910137
time: 1.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1683605, 0.1682937, -0.1201054, 0.1302733, -0.2986338, 0.2883991
1: -0.1163364, 0.1248079, -0.0878152, 0.0957153, -0.2120517, 0.2126231
2: -0.1081156, 0.1964567, -0.0896264, 0.1507547, -0.2588703, 0.2860831
3: 0.3592461, 1.0426173, 0.4865153, 1.0280288, -0.6687827, 0.5561020
4: -0.1365195, 0.1304664, -0.1043857, 0.1026190, -0.2391385, 0.2348522
5: -0.0667525, 0.5125632, -0.0472647, 0.4085015, -0.4752540, 0.5598279
6: -0.1437285, 0.1665697, -0.1142568, 0.1191907, -0.2629192, 0.2808266
7: -0.2112145, 0.1497268, -0.1580942, 0.1262557, -0.3374702, 0.3078210
8: -0.1165961, 0.1910026, -0.0665713, 0.1532523, -0.2698484, 0.2575738
9: -0.2464552, 0.1890608, -0.1930323, 0.1490031, -0.3954583, 0.3820931

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1683605, 0.1682937, -0.0961347, 0.1080595, -0.2764201, 0.2644284
1: -0.1163364, 0.1248079, -0.0727482, 0.0787322, -0.1950687, 0.1975561
2: -0.1081156, 0.1964567, -0.0782072, 0.1258768, -0.2339923, 0.2746639
3: 0.3592461, 1.0426173, 0.5510182, 1.0240104, -0.6647643, 0.4915991
4: -0.1365195, 0.1304664, -0.0873502, 0.0881315, -0.2246509, 0.2178167
5: -0.0667525, 0.5125632, -0.0397632, 0.3530774, -0.4198298, 0.5523264
6: -0.1437285, 0.1665697, -0.0979516, 0.0936279, -0.2373564, 0.2645213
7: -0.2112145, 0.1497268, -0.1322583, 0.1110893, -0.3223037, 0.2819852
8: -0.1165961, 0.1910026, -0.0453701, 0.1357013, -0.2522974, 0.2363727
9: -0.2464552, 0.1890608, -0.1688166, 0.1362091, -0.3826643, 0.3578774

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
time: 1.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7907137, upper bound: 0.7909472
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.1323535, 0.1398139, -0.2851998, 0.2865196
1: -0.1052670, 0.1127120, -0.0939614, 0.1055578, -0.2108248, 0.2066734
2: -0.1008987, 0.1787496, -0.0941783, 0.1624106, -0.2633093, 0.2729279
3: 0.4196393, 1.0340916, 0.4443746, 1.0336604, -0.6140211, 0.5897170
4: -0.1251058, 0.1190936, -0.1111998, 0.1083961, -0.2335018, 0.2302933
5: -0.0586379, 0.4646062, -0.0505697, 0.4460281, -0.5046660, 0.5151759
6: -0.1318853, 0.1461769, -0.1205043, 0.1328021, -0.2646874, 0.2666812
7: -0.1864930, 0.1413886, -0.1758752, 0.1307725, -0.3172656, 0.3172639
8: -0.0932197, 0.1716688, -0.0818439, 0.1581148, -0.2513345, 0.2535127
9: -0.2185586, 0.1716712, -0.2079326, 0.1538350, -0.3723936, 0.3796038

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8070467, upper bound: 0.7990571
time: 2.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8043608, upper bound: 0.7990100
time: 1.37 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.1186917, 0.1269170, -0.2723029, 0.2728577
1: -0.1052670, 0.1127120, -0.0852304, 0.0961525, -0.2014194, 0.1979424
2: -0.1008987, 0.1787496, -0.0878181, 0.1474300, -0.2483287, 0.2665677
3: 0.4196393, 1.0340916, 0.4772729, 1.0306065, -0.6109672, 0.5568187
4: -0.1251058, 0.1190936, -0.1003226, 0.0999454, -0.2250511, 0.2194162
5: -0.0586379, 0.4646062, -0.0457663, 0.4164358, -0.4750736, 0.5103725
6: -0.1318853, 0.1461769, -0.1106552, 0.1185563, -0.2504416, 0.2568322
7: -0.1864930, 0.1413886, -0.1618960, 0.1219839, -0.3084769, 0.3032847
8: -0.0932197, 0.1716688, -0.0691035, 0.1477058, -0.2409255, 0.2407724
9: -0.2185586, 0.1716712, -0.1941239, 0.1472560, -0.3658146, 0.3657951

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8070467, upper bound: 0.7990571
time: 1.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8043608, upper bound: 0.7990100
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1200036, 0.1301872, -0.1676116, 0.1666437, -0.2866472, 0.2977988
1: -0.0877545, 0.0956622, -0.1136909, 0.1257174, -0.2134720, 0.2093531
2: -0.0895770, 0.1506519, -0.1066725, 0.1935570, -0.2831340, 0.2573245
3: 0.4867636, 1.0279950, 0.3556925, 1.0427483, -0.5559847, 0.6723025
4: -0.1043288, 0.1025711, -0.1337138, 0.1287253, -0.2330541, 0.2362849
5: -0.0472382, 0.4083028, -0.0649723, 0.5173870, -0.5646252, 0.4732751
6: -0.1142021, 0.1190824, -0.1424800, 0.1651065, -0.2793085, 0.2615624
7: -0.1579918, 0.1261811, -0.2127165, 0.1462371, -0.3042288, 0.3388975
8: -0.0664715, 0.1531795, -0.1168307, 0.1890369, -0.2555085, 0.2700102
9: -0.1929342, 0.1489232, -0.2480972, 0.1849798, -0.3779140, 0.3970203

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1201054, 0.1302733, -0.1963227, 0.1938635, -0.3139689, 0.3265960
1: -0.0878152, 0.0957153, -0.1339435, 0.1477178, -0.2355330, 0.2296588
2: -0.0896264, 0.1507547, -0.1227878, 0.2224910, -0.3121175, 0.2735425
3: 0.4865153, 1.0280288, 0.2912362, 1.0473576, -0.5608422, 0.7367927
4: -0.1043857, 0.1026190, -0.1602450, 0.1484990, -0.2528847, 0.2628639
5: -0.0472647, 0.4085015, -0.0850894, 0.5667512, -0.6140160, 0.4935909
6: -0.1142568, 0.1191907, -0.1646160, 0.1921706, -0.3064275, 0.2838067
7: -0.1580942, 0.1262557, -0.2415616, 0.1629285, -0.3210227, 0.3678173
8: -0.0665713, 0.1532523, -0.1436460, 0.2210789, -0.2876502, 0.2968984
9: -0.1930323, 0.1490031, -0.2765329, 0.2212478, -0.4142802, 0.4255360

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7988673, upper bound: 0.7988727
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.1131819, 0.1219674, -0.2673533, 0.2673480
1: -0.1052670, 0.1127120, -0.0819143, 0.0920996, -0.1973665, 0.1946263
2: -0.1008987, 0.1787496, -0.0852243, 0.1422078, -0.2431066, 0.2639739
3: 0.4196393, 1.0340916, 0.4948269, 1.0293642, -0.6097249, 0.5392647
4: -0.1251058, 0.1190936, -0.0972459, 0.0972678, -0.2223736, 0.2163395
5: -0.0586379, 0.4646062, -0.0444142, 0.4013967, -0.4600345, 0.5090203
6: -0.1318853, 0.1461769, -0.1075819, 0.1124838, -0.2443692, 0.2537588
7: -0.1864930, 0.1413886, -0.1546251, 0.1193287, -0.3058218, 0.2960138
8: -0.0932197, 0.1716688, -0.0627786, 0.1446541, -0.2378738, 0.2344474
9: -0.2185586, 0.1716712, -0.1881767, 0.1446504, -0.3632090, 0.3598478

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8040244, upper bound: 0.7911140
time: 2.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8021425, upper bound: 0.7910632
time: 2.21 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1453859, 0.1541661, -0.1008312, 0.1093591, -0.2547450, 0.2549973
1: -0.1052670, 0.1127120, -0.0737615, 0.0830829, -0.1883499, 0.1864735
2: -0.1008987, 0.1787496, -0.0790291, 0.1297382, -0.2306369, 0.2577787
3: 0.4196393, 1.0340916, 0.5273200, 1.0266857, -0.6070464, 0.5067716
4: -0.1251058, 0.1190936, -0.0868989, 0.0889649, -0.2140707, 0.2059925
5: -0.0586379, 0.4646062, -0.0397120, 0.3735656, -0.4322035, 0.5043182
6: -0.1318853, 0.1461769, -0.0980690, 0.0989287, -0.2308140, 0.2442459
7: -0.1864930, 0.1413886, -0.1409962, 0.1107113, -0.2972043, 0.2823849
8: -0.0932197, 0.1716688, -0.0514807, 0.1345519, -0.2277716, 0.2231495
9: -0.2185586, 0.1716712, -0.1750948, 0.1379746, -0.3565333, 0.3467660

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8040244, upper bound: 0.7911140
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.8021425, upper bound: 0.7910632
time: 2.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1200036, 0.1301872, -0.1448753, 0.1472884, -0.2672919, 0.2750626
1: -0.0877545, 0.0956622, -0.0990862, 0.1126179, -0.2003725, 0.1947484
2: -0.0895770, 0.1506519, -0.0980847, 0.1718817, -0.2614587, 0.2487366
3: 0.4867636, 1.0279950, 0.4069002, 1.0387719, -0.5520083, 0.6210948
4: -0.1043288, 0.1025711, -0.1165836, 0.1136026, -0.2179314, 0.2191547
5: -0.0472382, 0.4083028, -0.0532438, 0.4745703, -0.5218084, 0.4615465
6: -0.1142021, 0.1190824, -0.1264503, 0.1444066, -0.2586086, 0.2455327
7: -0.1579918, 0.1261811, -0.1904205, 0.1353994, -0.2933912, 0.3166015
8: -0.0664715, 0.1531795, -0.0949170, 0.1676144, -0.2340859, 0.2480965
9: -0.1929342, 0.1489232, -0.2240686, 0.1639596, -0.3568938, 0.3729918

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909359
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909359
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1201054, 0.1302733, -0.1683605, 0.1682937, -0.2883991, 0.2986338
1: -0.0878152, 0.0957153, -0.1163364, 0.1248079, -0.2126231, 0.2120517
2: -0.0896264, 0.1507547, -0.1081156, 0.1964567, -0.2860831, 0.2588703
3: 0.4865153, 1.0280288, 0.3592461, 1.0426173, -0.5561020, 0.6687827
4: -0.1043857, 0.1026190, -0.1365195, 0.1304664, -0.2348522, 0.2391385
5: -0.0472647, 0.4085015, -0.0667525, 0.5125632, -0.5598279, 0.4752540
6: -0.1142568, 0.1191907, -0.1437285, 0.1665697, -0.2808266, 0.2629192
7: -0.1580942, 0.1262557, -0.2112145, 0.1497268, -0.3078210, 0.3374702
8: -0.0665713, 0.1532523, -0.1165961, 0.1910026, -0.2575738, 0.2698484
9: -0.1930323, 0.1490031, -0.2464552, 0.1890608, -0.3820931, 0.3954583

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909359
time: 1.92 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7976762, upper bound: 0.7909359
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1176214, 0.1280110, -0.1606673, 0.1615451, -0.2791666, 0.2886783
1: -0.0862550, 0.0938514, -0.1105849, 0.1220409, -0.2082959, 0.2044363
2: -0.0884515, 0.1485549, -0.1044699, 0.1884535, -0.2769049, 0.2530248
3: 0.4953572, 1.0276734, 0.3743249, 1.0392920, -0.5439347, 0.6533484
4: -0.1033042, 0.1016172, -0.1302291, 0.1253488, -0.2286530, 0.2318463
5: -0.0466779, 0.4008894, -0.0622286, 0.5033692, -0.5500470, 0.4631180
6: -0.1130880, 0.1162653, -0.1385837, 0.1592154, -0.2723034, 0.2548490
7: -0.1542481, 0.1253109, -0.2049520, 0.1439491, -0.2981972, 0.3302629
8: -0.0635725, 0.1520059, -0.1091712, 0.1822783, -0.2458508, 0.2611771
9: -0.1903824, 0.1479826, -0.2393888, 0.1774941, -0.3678764, 0.3873714

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928726
time: 1.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928726
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1176214, 0.1280110, -0.1442841, 0.1478183, -0.2654397, 0.2722951
1: -0.0862550, 0.0938514, -0.0994391, 0.1127605, -0.1990155, 0.1932905
2: -0.0884515, 0.1485549, -0.0983008, 0.1717905, -0.2602420, 0.2468558
3: 0.4953572, 1.0276734, 0.4094378, 1.0362738, -0.5409166, 0.6182356
4: -0.1033042, 0.1016172, -0.1164288, 0.1137928, -0.2170970, 0.2180460
5: -0.0466779, 0.4008894, -0.0530722, 0.4738035, -0.5204813, 0.4539616
6: -0.1130880, 0.1162653, -0.1267351, 0.1440900, -0.2571780, 0.2430004
7: -0.1542481, 0.1253109, -0.1893097, 0.1354625, -0.2897105, 0.3146206
8: -0.0635725, 0.1520059, -0.0933236, 0.1670191, -0.2305916, 0.2453295
9: -0.1903824, 0.1479826, -0.2224768, 0.1607780, -0.3511604, 0.3704594

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928726
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7963163, upper bound: 0.7928726
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0960476, 0.1079945, -0.1676116, 0.1666437, -0.2626913, 0.2756062
1: -0.0727019, 0.0786828, -0.1136909, 0.1257174, -0.1984194, 0.1923737
2: -0.0781674, 0.1257948, -0.1066725, 0.1935570, -0.2717243, 0.2324673
3: 0.5512632, 1.0239917, 0.3556925, 1.0427483, -0.4914851, 0.6682992
4: -0.0873080, 0.0880874, -0.1337138, 0.1287253, -0.2160333, 0.2218013
5: -0.0397462, 0.3528910, -0.0649723, 0.5173870, -0.5571333, 0.4178633
6: -0.0979064, 0.0935319, -0.1424800, 0.1651065, -0.2630128, 0.2360118
7: -0.1321757, 0.1110208, -0.2127165, 0.1462371, -0.2784127, 0.3237372
8: -0.0453001, 0.1356346, -0.1168307, 0.1890369, -0.2343371, 0.2524652
9: -0.1687368, 0.1361394, -0.2480972, 0.1849798, -0.3537166, 0.3842366

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0960476, 0.1079945, -0.1448753, 0.1472884, -0.2433360, 0.2528698
1: -0.0727019, 0.0786828, -0.0990862, 0.1126179, -0.1853199, 0.1777690
2: -0.0781674, 0.1257948, -0.0980847, 0.1718817, -0.2500491, 0.2238795
3: 0.5512632, 1.0239917, 0.4069002, 1.0387719, -0.4875087, 0.6170915
4: -0.0873080, 0.0880874, -0.1165836, 0.1136026, -0.2009106, 0.2046710
5: -0.0397462, 0.3528910, -0.0532438, 0.4745703, -0.5143165, 0.4061348
6: -0.0979064, 0.0935319, -0.1264503, 0.1444066, -0.2423130, 0.2199821
7: -0.1321757, 0.1110208, -0.1904205, 0.1353994, -0.2675751, 0.3014412
8: -0.0453001, 0.1356346, -0.0949170, 0.1676144, -0.2129145, 0.2305516
9: -0.1687368, 0.1361394, -0.2240686, 0.1639596, -0.3326964, 0.3602080

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.50 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7910137, upper bound: 0.7927558
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1177172, 0.1280896, -0.1896216, 0.1880180, -0.3057352, 0.3177112
1: -0.0863102, 0.0939018, -0.1318810, 0.1436484, -0.2299586, 0.2257828
2: -0.0884963, 0.1486507, -0.1208613, 0.2178894, -0.3063858, 0.2695119
3: 0.4951157, 1.0277025, 0.3108215, 1.0440311, -0.5489154, 0.7168809
4: -0.1033555, 0.1016615, -0.1572265, 0.1460620, -0.2494175, 0.2588879
5: -0.0467021, 0.4010797, -0.0834965, 0.5518676, -0.5985698, 0.4845762
6: -0.1131378, 0.1163656, -0.1608689, 0.1865107, -0.2996484, 0.2772344
7: -0.1543467, 0.1253784, -0.2335046, 0.1630585, -0.3174052, 0.3588830
8: -0.0636586, 0.1520712, -0.1364335, 0.2143974, -0.2780560, 0.2885047
9: -0.1904658, 0.1480583, -0.2678609, 0.2140354, -0.4045012, 0.4159192

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908369
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908369
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1177172, 0.1280896, -0.1692388, 0.1696041, -0.2873213, 0.2973284
1: -0.0863102, 0.0939018, -0.1171361, 0.1269905, -0.2133007, 0.2110379
2: -0.0884963, 0.1486507, -0.1095088, 0.1972732, -0.2857696, 0.2581595
3: 0.4951157, 1.0277025, 0.3563203, 1.0405133, -0.5453976, 0.6713821
4: -0.1033555, 0.1016615, -0.1374932, 0.1310381, -0.2343936, 0.2391547
5: -0.0467021, 0.4010797, -0.0672841, 0.5154582, -0.5621604, 0.4683637
6: -0.1131378, 0.1163656, -0.1449645, 0.1676442, -0.2807819, 0.2613301
7: -0.1543467, 0.1253784, -0.2123482, 0.1502874, -0.3046340, 0.3377266
8: -0.0636586, 0.1520712, -0.1165655, 0.1923174, -0.2559760, 0.2686367
9: -0.1904658, 0.1480583, -0.2467918, 0.1877160, -0.3781818, 0.3948502

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908369
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7962326, upper bound: 0.7908369
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0961347, 0.1080595, -0.1963874, 0.1938635, -0.2899982, 0.3044469
1: -0.0727482, 0.0787322, -0.1345346, 0.1477779, -0.2205261, 0.2132668
2: -0.0782072, 0.1258768, -0.1231195, 0.2225223, -0.3007296, 0.2489962
3: 0.5510182, 1.0240104, 0.2911931, 1.0473754, -0.4963573, 0.7328173
4: -0.0873502, 0.0881315, -0.1602945, 0.1489823, -0.2363326, 0.2484260
5: -0.0397632, 0.3530774, -0.0860005, 0.5667777, -0.6065409, 0.4390778
6: -0.0979516, 0.0936279, -0.1646202, 0.1922804, -0.2902320, 0.2582481
7: -0.1322583, 0.1110893, -0.2415916, 0.1649047, -0.2971630, 0.3526809
8: -0.0453701, 0.1357013, -0.1436770, 0.2211606, -0.2665307, 0.2793784
9: -0.1688166, 0.1362091, -0.2768697, 0.2212478, -0.3900644, 0.4130787

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7909472, upper bound: 0.7907137
time: 1.71 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.02 + 595.42 = 600.44 seconds
