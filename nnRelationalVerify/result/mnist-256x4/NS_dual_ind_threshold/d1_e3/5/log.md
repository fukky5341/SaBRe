## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000711535


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0031298, 0.0031298)
1: (-0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008824, 0.0008824)
2: (0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0065105, 0.0065105)
3: (0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008616, 0.0008616)
4: (-0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0048656, 0.0048656)
5: (0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013518, 0.0013518)
6: (0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012270, 0.0012270)
7: (-0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045790, 0.0045790)
8: (-0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035639, 0.0035639)
9: (-0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003075, 0.0003075)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 2.98 = 4.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0008371, upper bound: 0.0008363

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0007855
time: 2.12 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0008098
time: 1.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.05
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0007855
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.05
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0008098

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0039324, -0.0004647, -0.0040499, -0.0004476, -0.0030757, 0.0030513
1: -0.0040473, -0.0030697, -0.0040805, -0.0030649, -0.0008672, 0.0008603
2: 0.0086977, 0.0159112, 0.0084533, 0.0159468, -0.0063981, 0.0063473
3: 0.0027783, 0.0037329, 0.0027460, 0.0037376, -0.0008467, 0.0008400
4: -0.0057991, -0.0004083, -0.0058258, -0.0002256, -0.0047436, 0.0047816
5: 0.9938951, 0.9953928, 0.9938877, 0.9954436, -0.0013179, 0.0013285
6: 0.0023422, 0.0037017, 0.0023355, 0.0037478, -0.0011963, 0.0012058
7: -0.0146408, -0.0095674, -0.0146658, -0.0093955, -0.0044643, 0.0045000
8: -0.0017466, 0.0022021, -0.0018804, 0.0022216, -0.0035024, 0.0034745
9: -0.0041997, -0.0038591, -0.0042014, -0.0038475, -0.0002998, 0.0003022

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007864, upper bound: 0.0007864
time: 1.32 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007864, upper bound: 0.0007855
time: 2.14 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0039271, -0.0002519, -0.0040139, -0.0004537, -0.0030525, 0.0031318
1: -0.0040459, -0.0030097, -0.0040703, -0.0030666, -0.0008606, 0.0008830
2: 0.0087086, 0.0163538, 0.0085281, 0.0159340, -0.0063498, 0.0065148
3: 0.0027797, 0.0037915, 0.0027559, 0.0037359, -0.0008403, 0.0008621
4: -0.0061299, -0.0004164, -0.0058162, -0.0002815, -0.0048687, 0.0047454
5: 0.9938031, 0.9953905, 0.9938903, 0.9954280, -0.0013527, 0.0013184
6: 0.0022588, 0.0036997, 0.0023379, 0.0037337, -0.0012278, 0.0011967
7: -0.0149521, -0.0095750, -0.0146568, -0.0094481, -0.0045820, 0.0044660
8: -0.0017406, 0.0024444, -0.0018394, 0.0022146, -0.0034759, 0.0035662
9: -0.0042206, -0.0038596, -0.0042008, -0.0038510, -0.0003077, 0.0002999

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007691
time: 1.82 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007676, upper bound: 0.0007667
time: 1.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.84 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0007864, upper bound: 0.0007864
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0007864, upper bound: 0.0007855
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007691
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.84
Output dim: 5, lower bound: -0.0007676, upper bound: 0.0007667

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0039324, -0.0004647, -0.0039324, -0.0004647, -0.0029654, 0.0029654
1: -0.0040473, -0.0030697, -0.0040473, -0.0030697, -0.0008360, 0.0008360
2: 0.0086977, 0.0159112, 0.0086977, 0.0159112, -0.0061686, 0.0061686
3: 0.0027783, 0.0037329, 0.0027783, 0.0037329, -0.0008163, 0.0008163
4: -0.0057991, -0.0004083, -0.0057991, -0.0004083, -0.0046100, 0.0046100
5: 0.9938951, 0.9953928, 0.9938951, 0.9953928, -0.0012808, 0.0012808
6: 0.0023422, 0.0037017, 0.0023422, 0.0037017, -0.0011626, 0.0011626
7: -0.0146408, -0.0095674, -0.0146408, -0.0095674, -0.0043385, 0.0043385
8: -0.0017466, 0.0022021, -0.0017466, 0.0022021, -0.0033767, 0.0033767
9: -0.0041997, -0.0038591, -0.0041997, -0.0038591, -0.0002913, 0.0002913

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007579, upper bound: 0.0007396
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007578, upper bound: 0.0007488
time: 1.52 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0039324, -0.0004647, -0.0039271, -0.0002519, -0.0030961, 0.0029082
1: -0.0040473, -0.0030697, -0.0040459, -0.0030097, -0.0008729, 0.0008199
2: 0.0086977, 0.0159112, 0.0087086, 0.0163538, -0.0064406, 0.0060497
3: 0.0027783, 0.0037329, 0.0027797, 0.0037915, -0.0008523, 0.0008006
4: -0.0057991, -0.0004083, -0.0061299, -0.0004164, -0.0045212, 0.0048133
5: 0.9938951, 0.9953928, 0.9938031, 0.9953905, -0.0012561, 0.0013373
6: 0.0023422, 0.0037017, 0.0022588, 0.0036997, -0.0011402, 0.0012138
7: -0.0146408, -0.0095674, -0.0149521, -0.0095750, -0.0042549, 0.0045298
8: -0.0017466, 0.0022021, -0.0017406, 0.0024444, -0.0035256, 0.0033116
9: -0.0041997, -0.0038591, -0.0042206, -0.0038596, -0.0002857, 0.0003042

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007579, upper bound: 0.0007396
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007578, upper bound: 0.0007496
time: 2.17 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0039271, -0.0002519, -0.0039329, -0.0004647, -0.0030442, 0.0030576
1: -0.0040459, -0.0030097, -0.0040475, -0.0030697, -0.0008583, 0.0008621
2: 0.0087086, 0.0163538, 0.0086966, 0.0159111, -0.0063327, 0.0063604
3: 0.0027797, 0.0037915, 0.0027782, 0.0037329, -0.0008380, 0.0008417
4: -0.0061299, -0.0004164, -0.0057991, -0.0004074, -0.0047534, 0.0047326
5: 0.9938031, 0.9953905, 0.9938951, 0.9953930, -0.0013206, 0.0013149
6: 0.0022588, 0.0036997, 0.0023422, 0.0037019, -0.0011987, 0.0011935
7: -0.0149521, -0.0095750, -0.0146408, -0.0095666, -0.0044735, 0.0044539
8: -0.0017406, 0.0024444, -0.0017472, 0.0022021, -0.0034665, 0.0034817
9: -0.0042206, -0.0038596, -0.0041997, -0.0038590, -0.0003004, 0.0002991

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007596
time: 1.81 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007672
time: 1.85 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038770, -0.0002630, -0.0038462, -0.0002263, -0.0032271, 0.0030280
1: -0.0040317, -0.0030128, -0.0040230, -0.0030025, -0.0009098, 0.0008537
2: 0.0088130, 0.0163307, 0.0088770, 0.0164070, -0.0067130, 0.0062989
3: 0.0027936, 0.0037884, 0.0028020, 0.0037985, -0.0008884, 0.0008336
4: -0.0061127, -0.0004944, -0.0061697, -0.0005422, -0.0047074, 0.0050169
5: 0.9938080, 0.9953689, 0.9937921, 0.9953556, -0.0013079, 0.0013938
6: 0.0022631, 0.0036800, 0.0022488, 0.0036679, -0.0011871, 0.0012652
7: -0.0149359, -0.0096484, -0.0149895, -0.0096935, -0.0044302, 0.0047215
8: -0.0016835, 0.0024318, -0.0016484, 0.0024735, -0.0036747, 0.0034481
9: -0.0042195, -0.0038645, -0.0042231, -0.0038675, -0.0002975, 0.0003170

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007380, upper bound: 0.0007486
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007490, upper bound: 0.0007482
time: 1.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.99 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007579, upper bound: 0.0007396
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007578, upper bound: 0.0007488
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007579, upper bound: 0.0007396
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007578, upper bound: 0.0007496
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007596
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007672
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007380, upper bound: 0.0007486
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.99
Output dim: 5, lower bound: -0.0007490, upper bound: 0.0007482

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038522, -0.0004756, -0.0039324, -0.0004647, -0.0028929, 0.0029572
1: -0.0040248, -0.0030727, -0.0040473, -0.0030697, -0.0008156, 0.0008337
2: 0.0088644, 0.0158886, 0.0086977, 0.0159112, -0.0060179, 0.0061515
3: 0.0028004, 0.0037299, 0.0027783, 0.0037329, -0.0007964, 0.0008141
4: -0.0057823, -0.0005328, -0.0057991, -0.0004083, -0.0045972, 0.0044974
5: 0.9938998, 0.9953582, 0.9938951, 0.9953928, -0.0012773, 0.0012495
6: 0.0023465, 0.0036703, 0.0023422, 0.0037017, -0.0011594, 0.0011342
7: -0.0146249, -0.0096846, -0.0146408, -0.0095674, -0.0043265, 0.0042325
8: -0.0016553, 0.0021897, -0.0017466, 0.0022021, -0.0032942, 0.0033673
9: -0.0041987, -0.0038669, -0.0041997, -0.0038591, -0.0002905, 0.0002842

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007486
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007477
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037675, -0.0002353, -0.0038791, -0.0004756, -0.0028653, 0.0030283
1: -0.0040008, -0.0030050, -0.0040323, -0.0030728, -0.0008078, 0.0008538
2: 0.0090408, 0.0163884, 0.0088086, 0.0158885, -0.0059603, 0.0062995
3: 0.0028237, 0.0037960, 0.0027930, 0.0037299, -0.0007888, 0.0008336
4: -0.0061558, -0.0006646, -0.0057822, -0.0004911, -0.0047078, 0.0044544
5: 0.9937960, 0.9953216, 0.9938998, 0.9953698, -0.0013080, 0.0012376
6: 0.0022523, 0.0036371, 0.0023465, 0.0036808, -0.0011872, 0.0011233
7: -0.0149764, -0.0098086, -0.0146248, -0.0096453, -0.0044306, 0.0041921
8: -0.0015588, 0.0024633, -0.0016859, 0.0021896, -0.0032627, 0.0034483
9: -0.0042223, -0.0038753, -0.0041986, -0.0038643, -0.0002975, 0.0002815

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007427, upper bound: 0.0007312
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007427, upper bound: 0.0007418
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038522, -0.0004756, -0.0039271, -0.0002519, -0.0030237, 0.0029000
1: -0.0040248, -0.0030727, -0.0040459, -0.0030097, -0.0008525, 0.0008176
2: 0.0088644, 0.0158886, 0.0087086, 0.0163538, -0.0062899, 0.0060326
3: 0.0028004, 0.0037299, 0.0027797, 0.0037915, -0.0008324, 0.0007983
4: -0.0057823, -0.0005328, -0.0061299, -0.0004164, -0.0045084, 0.0047007
5: 0.9938998, 0.9953582, 0.9938031, 0.9953905, -0.0012526, 0.0013060
6: 0.0023465, 0.0036703, 0.0022588, 0.0036997, -0.0011370, 0.0011854
7: -0.0146249, -0.0096846, -0.0149521, -0.0095750, -0.0042429, 0.0044239
8: -0.0016553, 0.0021897, -0.0017406, 0.0024444, -0.0034431, 0.0033023
9: -0.0041987, -0.0038669, -0.0042206, -0.0038596, -0.0002849, 0.0002971

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007400
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007400
time: 2.10 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037675, -0.0002353, -0.0038770, -0.0002630, -0.0029953, 0.0029828
1: -0.0040008, -0.0030050, -0.0040317, -0.0030128, -0.0008445, 0.0008409
2: 0.0090408, 0.0163884, 0.0088130, 0.0163307, -0.0062309, 0.0062047
3: 0.0028237, 0.0037960, 0.0027936, 0.0037884, -0.0008246, 0.0008211
4: -0.0061558, -0.0006646, -0.0061127, -0.0004944, -0.0046370, 0.0046566
5: 0.9937960, 0.9953216, 0.9938080, 0.9953689, -0.0012883, 0.0012937
6: 0.0022523, 0.0036371, 0.0022631, 0.0036800, -0.0011694, 0.0011743
7: -0.0149764, -0.0098086, -0.0149359, -0.0096484, -0.0043640, 0.0043824
8: -0.0015588, 0.0024633, -0.0016835, 0.0024318, -0.0034108, 0.0033965
9: -0.0042223, -0.0038753, -0.0042195, -0.0038645, -0.0002930, 0.0002943

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007211
time: 1.82 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007491, upper bound: 0.0007302
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038517, -0.0002629, -0.0039329, -0.0004647, -0.0029759, 0.0030488
1: -0.0040246, -0.0030128, -0.0040475, -0.0030697, -0.0008390, 0.0008596
2: 0.0088655, 0.0163310, 0.0086966, 0.0159111, -0.0061905, 0.0063421
3: 0.0028005, 0.0037884, 0.0027782, 0.0037329, -0.0008192, 0.0008393
4: -0.0061129, -0.0005337, -0.0057991, -0.0004074, -0.0047397, 0.0046264
5: 0.9938079, 0.9953580, 0.9938951, 0.9953930, -0.0013168, 0.0012854
6: 0.0022631, 0.0036701, 0.0023422, 0.0037019, -0.0011953, 0.0011667
7: -0.0149361, -0.0096854, -0.0146408, -0.0095666, -0.0044606, 0.0043540
8: -0.0016547, 0.0024319, -0.0017472, 0.0022021, -0.0033887, 0.0034717
9: -0.0042195, -0.0038670, -0.0041997, -0.0038590, -0.0002995, 0.0002924

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007376
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007484
time: 1.92 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037664, -0.0000226, -0.0039329, -0.0004647, -0.0029008, 0.0031965
1: -0.0040006, -0.0029450, -0.0040475, -0.0030697, -0.0008178, 0.0009012
2: 0.0090429, 0.0168308, 0.0086966, 0.0159111, -0.0060343, 0.0066493
3: 0.0028240, 0.0038546, 0.0027782, 0.0037329, -0.0007985, 0.0008799
4: -0.0064864, -0.0006662, -0.0057991, -0.0004074, -0.0049693, 0.0045096
5: 0.9937041, 0.9953211, 0.9938951, 0.9953930, -0.0013806, 0.0012529
6: 0.0021689, 0.0036367, 0.0023422, 0.0037019, -0.0012532, 0.0011373
7: -0.0152876, -0.0098101, -0.0146408, -0.0095666, -0.0046767, 0.0042441
8: -0.0015576, 0.0027055, -0.0017472, 0.0022021, -0.0033032, 0.0036399
9: -0.0042432, -0.0038754, -0.0041997, -0.0038590, -0.0003140, 0.0002850

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007393
time: 1.95 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007504
time: 1.82 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038705, -0.0003872, -0.0038446, -0.0002635, -0.0030974, 0.0027921
1: -0.0040299, -0.0030478, -0.0040226, -0.0030129, -0.0008733, 0.0007872
2: 0.0088265, 0.0160724, 0.0088803, 0.0163297, -0.0064433, 0.0058081
3: 0.0027953, 0.0037542, 0.0028025, 0.0037883, -0.0008527, 0.0007686
4: -0.0059197, -0.0005045, -0.0061120, -0.0005447, -0.0043406, 0.0048153
5: 0.9938616, 0.9953661, 0.9938082, 0.9953550, -0.0012060, 0.0013378
6: 0.0023118, 0.0036774, 0.0022633, 0.0036673, -0.0010946, 0.0012144
7: -0.0147542, -0.0096579, -0.0149352, -0.0096958, -0.0040850, 0.0045318
8: -0.0016761, 0.0022904, -0.0016466, 0.0024312, -0.0035271, 0.0031794
9: -0.0042073, -0.0038651, -0.0042195, -0.0038677, -0.0002743, 0.0003043

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0007482
time: 1.86 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0007482
time: 2.15 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0039194, -0.0003599, -0.0038449, -0.0002602, -0.0031996, 0.0028184
1: -0.0040437, -0.0030401, -0.0040227, -0.0030120, -0.0009021, 0.0007946
2: 0.0087246, 0.0161291, 0.0088797, 0.0163365, -0.0066558, 0.0058629
3: 0.0027819, 0.0037617, 0.0028024, 0.0037892, -0.0008808, 0.0007759
4: -0.0059620, -0.0004284, -0.0061170, -0.0005442, -0.0043815, 0.0049741
5: 0.9938499, 0.9953873, 0.9938067, 0.9953551, -0.0012173, 0.0013820
6: 0.0023011, 0.0036966, 0.0022620, 0.0036674, -0.0011050, 0.0012544
7: -0.0147941, -0.0095863, -0.0149399, -0.0096953, -0.0041235, 0.0046812
8: -0.0017318, 0.0023214, -0.0016470, 0.0024349, -0.0036434, 0.0032093
9: -0.0042100, -0.0038603, -0.0042198, -0.0038676, -0.0002769, 0.0003143

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007486
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007305, upper bound: 0.0007486
time: 2.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.32 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007486
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007477
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007427, upper bound: 0.0007312
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007427, upper bound: 0.0007418
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007400
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007596, upper bound: 0.0007400
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007211
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007491, upper bound: 0.0007302
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007376
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007484
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007393
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007504
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0007482
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007211, upper bound: 0.0007482
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007486
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 5, lower bound: -0.0007305, upper bound: 0.0007486

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038522, -0.0004756, -0.0038522, -0.0004756, -0.0028847, 0.0028847
1: -0.0040248, -0.0030727, -0.0040248, -0.0030727, -0.0008133, 0.0008133
2: 0.0088644, 0.0158886, 0.0088644, 0.0158886, -0.0060008, 0.0060008
3: 0.0028004, 0.0037299, 0.0028004, 0.0037299, -0.0007941, 0.0007941
4: -0.0057823, -0.0005328, -0.0057823, -0.0005328, -0.0044846, 0.0044846
5: 0.9938998, 0.9953582, 0.9938998, 0.9953582, -0.0012460, 0.0012460
6: 0.0023465, 0.0036703, 0.0023465, 0.0036703, -0.0011310, 0.0011310
7: -0.0146249, -0.0096846, -0.0146249, -0.0096846, -0.0042205, 0.0042205
8: -0.0016553, 0.0021897, -0.0016553, 0.0021897, -0.0032848, 0.0032848
9: -0.0041987, -0.0038669, -0.0041987, -0.0038669, -0.0002834, 0.0002834

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007297
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007301
time: 2.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038522, -0.0004756, -0.0037675, -0.0002353, -0.0030040, 0.0028097
1: -0.0040248, -0.0030727, -0.0040008, -0.0030050, -0.0008470, 0.0007922
2: 0.0088644, 0.0158886, 0.0090408, 0.0163884, -0.0062490, 0.0058447
3: 0.0028004, 0.0037299, 0.0028237, 0.0037960, -0.0008270, 0.0007735
4: -0.0057823, -0.0005328, -0.0061558, -0.0006646, -0.0043680, 0.0046701
5: 0.9938998, 0.9953582, 0.9937960, 0.9953216, -0.0012136, 0.0012975
6: 0.0023465, 0.0036703, 0.0022523, 0.0036371, -0.0011015, 0.0011777
7: -0.0146249, -0.0096846, -0.0149764, -0.0098086, -0.0041108, 0.0043951
8: -0.0016553, 0.0021897, -0.0015588, 0.0024633, -0.0034207, 0.0031994
9: -0.0041987, -0.0038669, -0.0042223, -0.0038753, -0.0002760, 0.0002951

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007297
time: 2.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007297
time: 2.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037658, -0.0002720, -0.0038716, -0.0006034, -0.0027336, 0.0029781
1: -0.0040004, -0.0030154, -0.0040302, -0.0031088, -0.0007707, 0.0008396
2: 0.0090441, 0.0163120, 0.0088241, 0.0156226, -0.0056865, 0.0061951
3: 0.0028241, 0.0037859, 0.0027950, 0.0036947, -0.0007525, 0.0008198
4: -0.0060987, -0.0006672, -0.0055835, -0.0005027, -0.0046299, 0.0042498
5: 0.9938118, 0.9953209, 0.9939550, 0.9953666, -0.0012863, 0.0011807
6: 0.0022667, 0.0036364, 0.0023966, 0.0036779, -0.0011676, 0.0010717
7: -0.0149227, -0.0098110, -0.0144379, -0.0096563, -0.0043572, 0.0039995
8: -0.0015569, 0.0024215, -0.0016774, 0.0020441, -0.0031128, 0.0033912
9: -0.0042186, -0.0038754, -0.0041861, -0.0038650, -0.0002926, 0.0002686

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 114

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007191, upper bound: 0.0007118
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007122
time: 2.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037661, -0.0002687, -0.0039242, -0.0005731, -0.0027604, 0.0030827
1: -0.0040005, -0.0030144, -0.0040450, -0.0031002, -0.0007783, 0.0008691
2: 0.0090435, 0.0163190, 0.0087146, 0.0156856, -0.0057421, 0.0064126
3: 0.0028241, 0.0037868, 0.0027805, 0.0037030, -0.0007599, 0.0008486
4: -0.0061039, -0.0006667, -0.0056306, -0.0004209, -0.0047924, 0.0042913
5: 0.9938104, 0.9953210, 0.9939418, 0.9953893, -0.0013315, 0.0011923
6: 0.0022654, 0.0036365, 0.0023847, 0.0036985, -0.0012086, 0.0010822
7: -0.0149276, -0.0098106, -0.0144821, -0.0095793, -0.0045102, 0.0040386
8: -0.0015573, 0.0024253, -0.0017373, 0.0020786, -0.0031432, 0.0035103
9: -0.0042190, -0.0038754, -0.0041891, -0.0038598, -0.0003028, 0.0002712

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007196, upper bound: 0.0007208
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007216
time: 1.43 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038522, -0.0004756, -0.0038517, -0.0002629, -0.0030149, 0.0028290
1: -0.0040248, -0.0030727, -0.0040246, -0.0030128, -0.0008500, 0.0007976
2: 0.0088644, 0.0158886, 0.0088655, 0.0163310, -0.0062716, 0.0058848
3: 0.0028004, 0.0037299, 0.0028005, 0.0037884, -0.0008299, 0.0007788
4: -0.0057823, -0.0005328, -0.0061129, -0.0005337, -0.0043980, 0.0046870
5: 0.9938998, 0.9953582, 0.9938079, 0.9953580, -0.0012219, 0.0013022
6: 0.0023465, 0.0036703, 0.0022631, 0.0036701, -0.0011091, 0.0011820
7: -0.0146249, -0.0096846, -0.0149361, -0.0096854, -0.0041390, 0.0044110
8: -0.0016553, 0.0021897, -0.0016547, 0.0024319, -0.0034331, 0.0032214
9: -0.0041987, -0.0038669, -0.0042195, -0.0038670, -0.0002779, 0.0002962

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007236, upper bound: 0.0007209
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007488, upper bound: 0.0007209
time: 1.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038522, -0.0004756, -0.0037664, -0.0000226, -0.0031497, 0.0027559
1: -0.0040248, -0.0030727, -0.0040006, -0.0029450, -0.0008880, 0.0007770
2: 0.0088644, 0.0158886, 0.0090429, 0.0168308, -0.0065520, 0.0057329
3: 0.0028004, 0.0037299, 0.0028240, 0.0038546, -0.0008671, 0.0007587
4: -0.0057823, -0.0005328, -0.0064864, -0.0006662, -0.0042844, 0.0048966
5: 0.9938998, 0.9953582, 0.9937041, 0.9953211, -0.0011903, 0.0013604
6: 0.0023465, 0.0036703, 0.0021689, 0.0036367, -0.0010805, 0.0012348
7: -0.0146249, -0.0096846, -0.0152876, -0.0098101, -0.0040321, 0.0046082
8: -0.0016553, 0.0021897, -0.0015576, 0.0027055, -0.0035866, 0.0031382
9: -0.0041987, -0.0038669, -0.0042432, -0.0038754, -0.0002708, 0.0003094

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007236, upper bound: 0.0007217
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007488, upper bound: 0.0007213
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037658, -0.0002720, -0.0038705, -0.0003872, -0.0027465, 0.0029342
1: -0.0040004, -0.0030154, -0.0040299, -0.0030478, -0.0007743, 0.0008273
2: 0.0090441, 0.0163120, 0.0088265, 0.0160724, -0.0057132, 0.0061038
3: 0.0028241, 0.0037859, 0.0027953, 0.0037542, -0.0007561, 0.0008077
4: -0.0060987, -0.0006672, -0.0059197, -0.0005045, -0.0045616, 0.0042697
5: 0.9938118, 0.9953209, 0.9938616, 0.9953661, -0.0012674, 0.0011863
6: 0.0022667, 0.0036364, 0.0023118, 0.0036774, -0.0011504, 0.0010768
7: -0.0149227, -0.0098110, -0.0147542, -0.0096579, -0.0042930, 0.0040183
8: -0.0015569, 0.0024215, -0.0016761, 0.0022904, -0.0031274, 0.0033412
9: -0.0042186, -0.0038754, -0.0042073, -0.0038651, -0.0002883, 0.0002698

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007267, upper bound: 0.0007012
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007016
time: 1.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037661, -0.0002687, -0.0039194, -0.0003599, -0.0027728, 0.0030220
1: -0.0040005, -0.0030144, -0.0040437, -0.0030401, -0.0007818, 0.0008520
2: 0.0090435, 0.0163190, 0.0087246, 0.0161291, -0.0057680, 0.0062864
3: 0.0028241, 0.0037868, 0.0027819, 0.0037617, -0.0007633, 0.0008319
4: -0.0061039, -0.0006667, -0.0059620, -0.0004284, -0.0046981, 0.0043106
5: 0.9938104, 0.9953210, 0.9938499, 0.9953873, -0.0013053, 0.0011976
6: 0.0022654, 0.0036365, 0.0023011, 0.0036966, -0.0011848, 0.0010871
7: -0.0149276, -0.0098106, -0.0147941, -0.0095863, -0.0044214, 0.0040568
8: -0.0015573, 0.0024253, -0.0017318, 0.0023214, -0.0031574, 0.0034412
9: -0.0042190, -0.0038754, -0.0042100, -0.0038603, -0.0002969, 0.0002724

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 114

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007262, upper bound: 0.0007099
time: 1.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007099
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038498, -0.0002992, -0.0039255, -0.0005926, -0.0028431, 0.0030005
1: -0.0040241, -0.0030230, -0.0040454, -0.0031057, -0.0008016, 0.0008459
2: 0.0088694, 0.0162555, 0.0087121, 0.0156451, -0.0059143, 0.0062416
3: 0.0028010, 0.0037784, 0.0027802, 0.0036977, -0.0007827, 0.0008260
4: -0.0060564, -0.0005366, -0.0056003, -0.0004190, -0.0046645, 0.0044199
5: 0.9938236, 0.9953571, 0.9939502, 0.9953898, -0.0012959, 0.0012280
6: 0.0022773, 0.0036694, 0.0023923, 0.0036990, -0.0011763, 0.0011146
7: -0.0148829, -0.0096881, -0.0144537, -0.0095774, -0.0043899, 0.0041597
8: -0.0016526, 0.0023905, -0.0017387, 0.0020565, -0.0032375, 0.0034166
9: -0.0042160, -0.0038672, -0.0041872, -0.0038597, -0.0002948, 0.0002793

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007393, upper bound: 0.0007312
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007316
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038503, -0.0002968, -0.0039789, -0.0005615, -0.0028702, 0.0029814
1: -0.0040242, -0.0030223, -0.0040605, -0.0030970, -0.0008092, 0.0008406
2: 0.0088685, 0.0162604, 0.0086010, 0.0157099, -0.0059706, 0.0062018
3: 0.0028009, 0.0037791, 0.0027655, 0.0037063, -0.0007901, 0.0008207
4: -0.0060602, -0.0005359, -0.0056487, -0.0003360, -0.0046349, 0.0044621
5: 0.9938225, 0.9953574, 0.9939368, 0.9954129, -0.0012877, 0.0012397
6: 0.0022764, 0.0036695, 0.0023801, 0.0037199, -0.0011688, 0.0011253
7: -0.0148864, -0.0096875, -0.0144992, -0.0094993, -0.0043619, 0.0041993
8: -0.0016531, 0.0023933, -0.0017995, 0.0020919, -0.0032683, 0.0033949
9: -0.0042162, -0.0038671, -0.0041902, -0.0038545, -0.0002929, 0.0002820

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007397, upper bound: 0.0007411
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007415
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037649, -0.0000590, -0.0039255, -0.0005926, -0.0027688, 0.0031487
1: -0.0040001, -0.0029553, -0.0040454, -0.0031057, -0.0007806, 0.0008877
2: 0.0090460, 0.0167552, 0.0087121, 0.0156451, -0.0057596, 0.0065500
3: 0.0028244, 0.0038446, 0.0027802, 0.0036977, -0.0007622, 0.0008668
4: -0.0064299, -0.0006686, -0.0056003, -0.0004190, -0.0048951, 0.0043043
5: 0.9937198, 0.9953204, 0.9939502, 0.9953898, -0.0013600, 0.0011959
6: 0.0021831, 0.0036361, 0.0023923, 0.0036990, -0.0012345, 0.0010855
7: -0.0152344, -0.0098123, -0.0144537, -0.0095774, -0.0046068, 0.0040509
8: -0.0015559, 0.0026641, -0.0017387, 0.0020565, -0.0031528, 0.0035855
9: -0.0042396, -0.0038755, -0.0041872, -0.0038597, -0.0003093, 0.0002720

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007208
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007208
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037652, -0.0000562, -0.0039789, -0.0005615, -0.0027956, 0.0032452
1: -0.0040002, -0.0029545, -0.0040605, -0.0030970, -0.0007882, 0.0009149
2: 0.0090454, 0.0167610, 0.0086010, 0.0157099, -0.0058153, 0.0067507
3: 0.0028243, 0.0038453, 0.0027655, 0.0037063, -0.0007696, 0.0008933
4: -0.0064343, -0.0006681, -0.0056487, -0.0003360, -0.0050450, 0.0043460
5: 0.9937186, 0.9953206, 0.9939368, 0.9954129, -0.0014017, 0.0012075
6: 0.0021820, 0.0036362, 0.0023801, 0.0037199, -0.0012723, 0.0010960
7: -0.0152385, -0.0098119, -0.0144992, -0.0094993, -0.0047479, 0.0040901
8: -0.0015562, 0.0026673, -0.0017995, 0.0020919, -0.0031833, 0.0036953
9: -0.0042399, -0.0038755, -0.0041902, -0.0038545, -0.0003188, 0.0002746

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007309
time: 1.43 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007196, upper bound: 0.0007305
time: 1.99 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038705, -0.0003872, -0.0037658, -0.0002720, -0.0029342, 0.0027465
1: -0.0040299, -0.0030478, -0.0040004, -0.0030154, -0.0008273, 0.0007743
2: 0.0088265, 0.0160724, 0.0090441, 0.0163120, -0.0061038, 0.0057132
3: 0.0027953, 0.0037542, 0.0028241, 0.0037859, -0.0008077, 0.0007561
4: -0.0059197, -0.0005045, -0.0060987, -0.0006672, -0.0042697, 0.0045616
5: 0.9938616, 0.9953661, 0.9938118, 0.9953209, -0.0011863, 0.0012674
6: 0.0023118, 0.0036774, 0.0022667, 0.0036364, -0.0010768, 0.0011504
7: -0.0147542, -0.0096579, -0.0149227, -0.0098110, -0.0040183, 0.0042930
8: -0.0016761, 0.0022904, -0.0015569, 0.0024215, -0.0033412, 0.0031274
9: -0.0042073, -0.0038651, -0.0042186, -0.0038754, -0.0002698, 0.0002883

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006970, upper bound: 0.0007284
time: 1.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007284
time: 2.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038705, -0.0003872, -0.0037649, -0.0000580, -0.0030931, 0.0027155
1: -0.0040299, -0.0030478, -0.0040001, -0.0029550, -0.0008721, 0.0007656
2: 0.0088265, 0.0160724, 0.0090460, 0.0167572, -0.0064343, 0.0056487
3: 0.0027953, 0.0037542, 0.0028244, 0.0038448, -0.0008515, 0.0007475
4: -0.0059197, -0.0005045, -0.0064314, -0.0006686, -0.0042215, 0.0048086
5: 0.9938616, 0.9953661, 0.9937194, 0.9953204, -0.0011729, 0.0013360
6: 0.0023118, 0.0036774, 0.0021828, 0.0036361, -0.0010646, 0.0012127
7: -0.0147542, -0.0096579, -0.0152358, -0.0098123, -0.0039729, 0.0045255
8: -0.0016761, 0.0022904, -0.0015559, 0.0026652, -0.0035222, 0.0030921
9: -0.0042073, -0.0038651, -0.0042397, -0.0038755, -0.0002668, 0.0003039

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007284
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007020, upper bound: 0.0007280
time: 2.19 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0039194, -0.0003599, -0.0037661, -0.0002687, -0.0030220, 0.0027728
1: -0.0040437, -0.0030401, -0.0040005, -0.0030144, -0.0008520, 0.0007818
2: 0.0087246, 0.0161291, 0.0090435, 0.0163190, -0.0062864, 0.0057680
3: 0.0027819, 0.0037617, 0.0028241, 0.0037868, -0.0008319, 0.0007633
4: -0.0059620, -0.0004284, -0.0061039, -0.0006667, -0.0043106, 0.0046981
5: 0.9938499, 0.9953873, 0.9938104, 0.9953210, -0.0011976, 0.0013053
6: 0.0023011, 0.0036966, 0.0022654, 0.0036365, -0.0010871, 0.0011848
7: -0.0147941, -0.0095863, -0.0149276, -0.0098106, -0.0040568, 0.0044214
8: -0.0017318, 0.0023214, -0.0015573, 0.0024253, -0.0034412, 0.0031574
9: -0.0042100, -0.0038603, -0.0042190, -0.0038754, -0.0002724, 0.0002969

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0007284
time: 1.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007280
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0039194, -0.0003599, -0.0037652, -0.0000552, -0.0031945, 0.0027457
1: -0.0040437, -0.0030401, -0.0040002, -0.0029542, -0.0009006, 0.0007741
2: 0.0087246, 0.0161291, 0.0090454, 0.0167630, -0.0066452, 0.0057115
3: 0.0027819, 0.0037617, 0.0028243, 0.0038456, -0.0008794, 0.0007558
4: -0.0059620, -0.0004284, -0.0064357, -0.0006681, -0.0042684, 0.0049662
5: 0.9938499, 0.9953873, 0.9937181, 0.9953206, -0.0011859, 0.0013798
6: 0.0023011, 0.0036966, 0.0021817, 0.0036362, -0.0010764, 0.0012524
7: -0.0147941, -0.0095863, -0.0152399, -0.0098119, -0.0040171, 0.0046737
8: -0.0017318, 0.0023214, -0.0015562, 0.0026684, -0.0036376, 0.0031265
9: -0.0042100, -0.0038603, -0.0042399, -0.0038755, -0.0002697, 0.0003138

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0007280
time: 2.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007280
time: 2.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.59 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007297
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007301
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007297
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007297
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007191, upper bound: 0.0007118
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007122
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007196, upper bound: 0.0007208
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007216
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007236, upper bound: 0.0007209
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007488, upper bound: 0.0007209
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007236, upper bound: 0.0007217
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007488, upper bound: 0.0007213
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007267, upper bound: 0.0007012
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007016
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007262, upper bound: 0.0007099
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007099
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007393, upper bound: 0.0007312
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007316
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007397, upper bound: 0.0007411
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007415
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007208
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007208
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007309
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007196, upper bound: 0.0007305
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0006970, upper bound: 0.0007284
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007284
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007284
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007020, upper bound: 0.0007280
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0007284
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007280
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0007280
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007280

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038446, -0.0006034, -0.0038501, -0.0005127, -0.0028355, 0.0027524
1: -0.0040226, -0.0031088, -0.0040242, -0.0030832, -0.0007994, 0.0007760
2: 0.0088802, 0.0156227, 0.0088688, 0.0158114, -0.0058985, 0.0057256
3: 0.0028025, 0.0036947, 0.0028009, 0.0037197, -0.0007806, 0.0007577
4: -0.0055836, -0.0005447, -0.0057246, -0.0005361, -0.0042789, 0.0044082
5: 0.9939550, 0.9953550, 0.9939158, 0.9953573, -0.0011888, 0.0012247
6: 0.0023966, 0.0036673, 0.0023610, 0.0036695, -0.0010791, 0.0011117
7: -0.0144379, -0.0096957, -0.0145706, -0.0096877, -0.0040270, 0.0041486
8: -0.0016467, 0.0020442, -0.0016529, 0.0021475, -0.0032289, 0.0031342
9: -0.0041861, -0.0038677, -0.0041950, -0.0038671, -0.0002704, 0.0002786

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007225
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007166, upper bound: 0.0007249
time: 2.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0038998, -0.0005726, -0.0038507, -0.0005092, -0.0029426, 0.0027797
1: -0.0040382, -0.0031001, -0.0040243, -0.0030822, -0.0008296, 0.0007837
2: 0.0087655, 0.0156867, 0.0088676, 0.0158185, -0.0061213, 0.0057823
3: 0.0027873, 0.0037032, 0.0028008, 0.0037206, -0.0008101, 0.0007652
4: -0.0056314, -0.0004589, -0.0057299, -0.0005352, -0.0043213, 0.0045747
5: 0.9939417, 0.9953788, 0.9939143, 0.9953575, -0.0012006, 0.0012710
6: 0.0023845, 0.0036889, 0.0023597, 0.0036697, -0.0010898, 0.0011537
7: -0.0144829, -0.0096150, -0.0145756, -0.0096868, -0.0040668, 0.0043053
8: -0.0017095, 0.0020792, -0.0016536, 0.0021514, -0.0033508, 0.0031652
9: -0.0041891, -0.0038622, -0.0041953, -0.0038671, -0.0002731, 0.0002891

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007234
time: 2.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007258, upper bound: 0.0007249
time: 2.20 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038446, -0.0006034, -0.0037658, -0.0002720, -0.0029536, 0.0026783
1: -0.0040226, -0.0031088, -0.0040004, -0.0030154, -0.0008327, 0.0007551
2: 0.0088802, 0.0156227, 0.0090441, 0.0163120, -0.0061440, 0.0055713
3: 0.0028025, 0.0036947, 0.0028241, 0.0037859, -0.0008131, 0.0007373
4: -0.0055836, -0.0005447, -0.0060987, -0.0006672, -0.0041637, 0.0045917
5: 0.9939550, 0.9953550, 0.9938118, 0.9953209, -0.0011568, 0.0012757
6: 0.0023966, 0.0036673, 0.0022667, 0.0036364, -0.0010500, 0.0011580
7: -0.0144379, -0.0096957, -0.0149227, -0.0098110, -0.0039185, 0.0043213
8: -0.0016467, 0.0020442, -0.0015569, 0.0024215, -0.0033633, 0.0030497
9: -0.0041861, -0.0038677, -0.0042186, -0.0038754, -0.0002631, 0.0002902

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007074
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007093
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038998, -0.0005726, -0.0037661, -0.0002687, -0.0030617, 0.0027051
1: -0.0040382, -0.0031001, -0.0040005, -0.0030144, -0.0008632, 0.0007627
2: 0.0087655, 0.0156867, 0.0090435, 0.0163190, -0.0063690, 0.0056272
3: 0.0027873, 0.0037032, 0.0028241, 0.0037868, -0.0008428, 0.0007447
4: -0.0056314, -0.0004589, -0.0061039, -0.0006667, -0.0042054, 0.0047598
5: 0.9939417, 0.9953788, 0.9938104, 0.9953210, -0.0011684, 0.0013224
6: 0.0023845, 0.0036889, 0.0022654, 0.0036365, -0.0010605, 0.0012004
7: -0.0144829, -0.0096150, -0.0149276, -0.0098106, -0.0039578, 0.0044795
8: -0.0017095, 0.0020792, -0.0015573, 0.0024253, -0.0034864, 0.0030803
9: -0.0041891, -0.0038622, -0.0042190, -0.0038754, -0.0002658, 0.0003008

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007074
time: 2.08 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007096
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0037582, -0.0003802, -0.0038698, -0.0006320, -0.0026840, 0.0028773
1: -0.0039982, -0.0030459, -0.0040297, -0.0031168, -0.0007567, 0.0008112
2: 0.0090600, 0.0160869, 0.0088280, 0.0155632, -0.0055832, 0.0059855
3: 0.0028262, 0.0037561, 0.0027955, 0.0036868, -0.0007388, 0.0007921
4: -0.0059305, -0.0006790, -0.0055391, -0.0005056, -0.0044732, 0.0041725
5: 0.9938586, 0.9953176, 0.9939673, 0.9953658, -0.0012428, 0.0011593
6: 0.0023091, 0.0036334, 0.0024078, 0.0036772, -0.0011281, 0.0010522
7: -0.0147644, -0.0098222, -0.0143960, -0.0096590, -0.0042097, 0.0039268
8: -0.0015483, 0.0022983, -0.0016753, 0.0020116, -0.0030562, 0.0032764
9: -0.0042080, -0.0038762, -0.0041833, -0.0038652, -0.0002827, 0.0002637

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007118
time: 1.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007115
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037889, -0.0003642, -0.0038697, -0.0006335, -0.0027387, 0.0028920
1: -0.0040069, -0.0030413, -0.0040297, -0.0031173, -0.0007721, 0.0008154
2: 0.0089961, 0.0161203, 0.0088280, 0.0155601, -0.0056970, 0.0060160
3: 0.0028178, 0.0037606, 0.0027955, 0.0036864, -0.0007539, 0.0007961
4: -0.0059554, -0.0006313, -0.0055368, -0.0005056, -0.0044960, 0.0042576
5: 0.9938516, 0.9953309, 0.9939680, 0.9953657, -0.0012491, 0.0011829
6: 0.0023028, 0.0036455, 0.0024084, 0.0036772, -0.0011338, 0.0010737
7: -0.0147879, -0.0097772, -0.0143939, -0.0096590, -0.0042313, 0.0040069
8: -0.0015832, 0.0023166, -0.0016752, 0.0020099, -0.0031185, 0.0032932
9: -0.0042096, -0.0038731, -0.0041831, -0.0038652, -0.0002841, 0.0002691

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007096, upper bound: 0.0007118
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007096, upper bound: 0.0007118
time: 2.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0037586, -0.0003767, -0.0039224, -0.0006009, -0.0027112, 0.0029837
1: -0.0039983, -0.0030449, -0.0040445, -0.0031081, -0.0007644, 0.0008412
2: 0.0090593, 0.0160943, 0.0087184, 0.0156279, -0.0056399, 0.0062067
3: 0.0028261, 0.0037571, 0.0027810, 0.0036954, -0.0007463, 0.0008214
4: -0.0059360, -0.0006785, -0.0055874, -0.0004237, -0.0046385, 0.0042149
5: 0.9938570, 0.9953178, 0.9939539, 0.9953886, -0.0012887, 0.0011710
6: 0.0023077, 0.0036336, 0.0023956, 0.0036978, -0.0011698, 0.0010629
7: -0.0147696, -0.0098217, -0.0144415, -0.0095819, -0.0043654, 0.0039667
8: -0.0015486, 0.0023023, -0.0017353, 0.0020470, -0.0030873, 0.0033976
9: -0.0042084, -0.0038761, -0.0041863, -0.0038600, -0.0002931, 0.0002664

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007208
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007074, upper bound: 0.0007208
time: 2.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037892, -0.0003615, -0.0039224, -0.0006040, -0.0026327, 0.0029982
1: -0.0040070, -0.0030406, -0.0040445, -0.0031089, -0.0007423, 0.0008453
2: 0.0089955, 0.0161258, 0.0087185, 0.0156215, -0.0054766, 0.0062369
3: 0.0028177, 0.0037613, 0.0027811, 0.0036946, -0.0007247, 0.0008254
4: -0.0059596, -0.0006308, -0.0055827, -0.0004238, -0.0046611, 0.0040929
5: 0.9938505, 0.9953310, 0.9939551, 0.9953884, -0.0012950, 0.0011371
6: 0.0023018, 0.0036456, 0.0023968, 0.0036978, -0.0011755, 0.0010322
7: -0.0147918, -0.0097768, -0.0144371, -0.0095820, -0.0043866, 0.0038518
8: -0.0015836, 0.0023196, -0.0017352, 0.0020435, -0.0029979, 0.0034141
9: -0.0042099, -0.0038731, -0.0041860, -0.0038600, -0.0002946, 0.0002586

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007208
time: 1.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007208
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038446, -0.0006034, -0.0038498, -0.0002992, -0.0029662, 0.0026971
1: -0.0040226, -0.0031088, -0.0040241, -0.0030230, -0.0008363, 0.0007604
2: 0.0088802, 0.0156227, 0.0088694, 0.0162555, -0.0061702, 0.0056106
3: 0.0028025, 0.0036947, 0.0028010, 0.0037784, -0.0008165, 0.0007425
4: -0.0055836, -0.0005447, -0.0060564, -0.0005366, -0.0041930, 0.0046112
5: 0.9939550, 0.9953550, 0.9938236, 0.9953571, -0.0011649, 0.0012811
6: 0.0023966, 0.0036673, 0.0022773, 0.0036694, -0.0010574, 0.0011629
7: -0.0144379, -0.0096957, -0.0148829, -0.0096881, -0.0039461, 0.0043397
8: -0.0016467, 0.0020442, -0.0016526, 0.0023905, -0.0033776, 0.0030712
9: -0.0041861, -0.0038677, -0.0042160, -0.0038672, -0.0002650, 0.0002914

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007316, upper bound: 0.0007143
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007316, upper bound: 0.0007173
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0038998, -0.0005726, -0.0038503, -0.0002968, -0.0029420, 0.0027242
1: -0.0040382, -0.0031001, -0.0040242, -0.0030223, -0.0008295, 0.0007680
2: 0.0087655, 0.0156867, 0.0088685, 0.0162604, -0.0061201, 0.0056668
3: 0.0027873, 0.0037032, 0.0028009, 0.0037791, -0.0008099, 0.0007499
4: -0.0056314, -0.0004589, -0.0060602, -0.0005359, -0.0042350, 0.0045737
5: 0.9939417, 0.9953788, 0.9938225, 0.9953574, -0.0011766, 0.0012707
6: 0.0023845, 0.0036889, 0.0022764, 0.0036695, -0.0010680, 0.0011534
7: -0.0144829, -0.0096150, -0.0148864, -0.0096875, -0.0039856, 0.0043044
8: -0.0017095, 0.0020792, -0.0016531, 0.0023933, -0.0033501, 0.0031020
9: -0.0041891, -0.0038622, -0.0042162, -0.0038671, -0.0002676, 0.0002890

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007142
time: 1.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007164
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038446, -0.0006034, -0.0037649, -0.0000590, -0.0031014, 0.0026248
1: -0.0040226, -0.0031088, -0.0040001, -0.0029553, -0.0008744, 0.0007400
2: 0.0088802, 0.0156227, 0.0090460, 0.0167552, -0.0064516, 0.0054600
3: 0.0028025, 0.0036947, 0.0028244, 0.0038446, -0.0008538, 0.0007225
4: -0.0055836, -0.0005447, -0.0064299, -0.0006686, -0.0040805, 0.0048215
5: 0.9939550, 0.9953550, 0.9937198, 0.9953204, -0.0011337, 0.0013396
6: 0.0023966, 0.0036673, 0.0021831, 0.0036361, -0.0010290, 0.0012159
7: -0.0144379, -0.0096957, -0.0152344, -0.0098123, -0.0038402, 0.0045376
8: -0.0016467, 0.0020442, -0.0015559, 0.0026641, -0.0035316, 0.0029888
9: -0.0041861, -0.0038677, -0.0042396, -0.0038755, -0.0002579, 0.0003047

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007208, upper bound: 0.0006993
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007009
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038998, -0.0005726, -0.0037652, -0.0000562, -0.0032059, 0.0026516
1: -0.0040382, -0.0031001, -0.0040002, -0.0029545, -0.0009039, 0.0007476
2: 0.0087655, 0.0156867, 0.0090454, 0.0167610, -0.0066689, 0.0055158
3: 0.0027873, 0.0037032, 0.0028243, 0.0038453, -0.0008825, 0.0007299
4: -0.0056314, -0.0004589, -0.0064343, -0.0006681, -0.0041221, 0.0049839
5: 0.9939417, 0.9953788, 0.9937186, 0.9953206, -0.0011453, 0.0013847
6: 0.0023845, 0.0036889, 0.0021820, 0.0036362, -0.0010395, 0.0012569
7: -0.0144829, -0.0096150, -0.0152385, -0.0098119, -0.0038794, 0.0046904
8: -0.0017095, 0.0020792, -0.0015562, 0.0026673, -0.0036506, 0.0030193
9: -0.0041891, -0.0038622, -0.0042399, -0.0038755, -0.0002605, 0.0003150

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007309, upper bound: 0.0006993
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007309, upper bound: 0.0007017
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0037582, -0.0003802, -0.0038687, -0.0004145, -0.0027101, 0.0028334
1: -0.0039982, -0.0030459, -0.0040294, -0.0030555, -0.0007641, 0.0007988
2: 0.0090600, 0.0160869, 0.0088303, 0.0160156, -0.0056375, 0.0058941
3: 0.0028262, 0.0037561, 0.0027958, 0.0037467, -0.0007460, 0.0007800
4: -0.0059305, -0.0006790, -0.0058772, -0.0005073, -0.0044049, 0.0042131
5: 0.9938586, 0.9953176, 0.9938734, 0.9953653, -0.0012238, 0.0011705
6: 0.0023091, 0.0036334, 0.0023225, 0.0036767, -0.0011108, 0.0010625
7: -0.0147644, -0.0098222, -0.0147142, -0.0096606, -0.0041455, 0.0039650
8: -0.0015483, 0.0022983, -0.0016740, 0.0022592, -0.0030860, 0.0032264
9: -0.0042080, -0.0038762, -0.0042047, -0.0038653, -0.0002784, 0.0002662

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007062, upper bound: 0.0007012
time: 2.02 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007197, upper bound: 0.0007012
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037889, -0.0003642, -0.0038686, -0.0004156, -0.0027526, 0.0028481
1: -0.0040069, -0.0030413, -0.0040294, -0.0030558, -0.0007761, 0.0008030
2: 0.0089961, 0.0161203, 0.0088304, 0.0160133, -0.0057259, 0.0059246
3: 0.0028178, 0.0037606, 0.0027959, 0.0037464, -0.0007577, 0.0007840
4: -0.0059554, -0.0006313, -0.0058755, -0.0005074, -0.0044277, 0.0042792
5: 0.9938516, 0.9953309, 0.9938739, 0.9953653, -0.0012301, 0.0011889
6: 0.0023028, 0.0036455, 0.0023230, 0.0036767, -0.0011166, 0.0010792
7: -0.0147879, -0.0097772, -0.0147126, -0.0096607, -0.0041670, 0.0040272
8: -0.0015832, 0.0023166, -0.0016740, 0.0022580, -0.0031344, 0.0032432
9: -0.0042096, -0.0038731, -0.0042045, -0.0038653, -0.0002798, 0.0002704

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007012
time: 1.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007084, upper bound: 0.0007017
time: 2.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0037586, -0.0003767, -0.0039176, -0.0003865, -0.0027368, 0.0029230
1: -0.0039983, -0.0030449, -0.0040432, -0.0030476, -0.0007716, 0.0008241
2: 0.0090593, 0.0160943, 0.0087284, 0.0160738, -0.0056931, 0.0060804
3: 0.0028261, 0.0037571, 0.0027824, 0.0037544, -0.0007534, 0.0008046
4: -0.0059360, -0.0006785, -0.0059207, -0.0004312, -0.0045441, 0.0042546
5: 0.9938570, 0.9953178, 0.9938613, 0.9953865, -0.0012625, 0.0011821
6: 0.0023077, 0.0036336, 0.0023116, 0.0036959, -0.0011460, 0.0010730
7: -0.0147696, -0.0098217, -0.0147551, -0.0095889, -0.0042765, 0.0040041
8: -0.0015486, 0.0023023, -0.0017298, 0.0022911, -0.0031164, 0.0033284
9: -0.0042084, -0.0038761, -0.0042074, -0.0038605, -0.0002872, 0.0002689

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007103
time: 1.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007099
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037892, -0.0003615, -0.0039175, -0.0003892, -0.0027784, 0.0029375
1: -0.0040070, -0.0030406, -0.0040432, -0.0030484, -0.0007833, 0.0008282
2: 0.0089955, 0.0161258, 0.0087286, 0.0160681, -0.0057796, 0.0061105
3: 0.0028177, 0.0037613, 0.0027824, 0.0037537, -0.0007648, 0.0008086
4: -0.0059596, -0.0006308, -0.0059165, -0.0004314, -0.0045666, 0.0043193
5: 0.9938505, 0.9953310, 0.9938625, 0.9953864, -0.0012687, 0.0012000
6: 0.0023018, 0.0036456, 0.0023126, 0.0036959, -0.0011516, 0.0010893
7: -0.0147918, -0.0097768, -0.0147512, -0.0095891, -0.0042977, 0.0040650
8: -0.0015836, 0.0023196, -0.0017297, 0.0022880, -0.0031638, 0.0033449
9: -0.0042099, -0.0038731, -0.0042071, -0.0038605, -0.0002886, 0.0002730

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007095
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007095
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038425, -0.0004058, -0.0039236, -0.0006211, -0.0027935, 0.0027779
1: -0.0040220, -0.0030531, -0.0040449, -0.0031138, -0.0007876, 0.0007832
2: 0.0088847, 0.0160338, 0.0087159, 0.0155859, -0.0058111, 0.0057786
3: 0.0028030, 0.0037491, 0.0027807, 0.0036898, -0.0007690, 0.0007647
4: -0.0058908, -0.0005480, -0.0055561, -0.0004218, -0.0043186, 0.0043429
5: 0.9938695, 0.9953540, 0.9939626, 0.9953890, -0.0011998, 0.0012066
6: 0.0023191, 0.0036665, 0.0024035, 0.0036983, -0.0010891, 0.0010952
7: -0.0147270, -0.0096989, -0.0144120, -0.0095801, -0.0040643, 0.0040871
8: -0.0016442, 0.0022692, -0.0017366, 0.0020240, -0.0031810, 0.0031632
9: -0.0042055, -0.0038679, -0.0041844, -0.0038599, -0.0002729, 0.0002744

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007320
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007316
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0038727, -0.0003929, -0.0039236, -0.0006227, -0.0028493, 0.0027914
1: -0.0040305, -0.0030494, -0.0040449, -0.0031142, -0.0008033, 0.0007870
2: 0.0088218, 0.0160606, 0.0087159, 0.0155825, -0.0059272, 0.0058067
3: 0.0027947, 0.0037527, 0.0027807, 0.0036894, -0.0007844, 0.0007684
4: -0.0059108, -0.0005010, -0.0055535, -0.0004218, -0.0043396, 0.0044296
5: 0.9938640, 0.9953670, 0.9939633, 0.9953891, -0.0012057, 0.0012307
6: 0.0023140, 0.0036783, 0.0024042, 0.0036983, -0.0010944, 0.0011171
7: -0.0147459, -0.0096546, -0.0144096, -0.0095801, -0.0040840, 0.0041687
8: -0.0016787, 0.0022839, -0.0017366, 0.0020222, -0.0032445, 0.0031786
9: -0.0042068, -0.0038649, -0.0041842, -0.0038599, -0.0002742, 0.0002799

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0007312
time: 1.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0007312
time: 2.22 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038430, -0.0004021, -0.0039771, -0.0005891, -0.0028211, 0.0028800
1: -0.0040221, -0.0030520, -0.0040599, -0.0031048, -0.0007954, 0.0008120
2: 0.0088837, 0.0160414, 0.0086047, 0.0156523, -0.0058686, 0.0059909
3: 0.0028029, 0.0037501, 0.0027660, 0.0036986, -0.0007766, 0.0007928
4: -0.0058965, -0.0005473, -0.0056057, -0.0003388, -0.0044773, 0.0043858
5: 0.9938680, 0.9953542, 0.9939489, 0.9954121, -0.0012439, 0.0012185
6: 0.0023177, 0.0036667, 0.0023910, 0.0037192, -0.0011291, 0.0011060
7: -0.0147324, -0.0096982, -0.0144587, -0.0095019, -0.0042136, 0.0041275
8: -0.0016448, 0.0022734, -0.0017975, 0.0020604, -0.0032125, 0.0032794
9: -0.0042059, -0.0038678, -0.0041875, -0.0038547, -0.0002829, 0.0002772

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007407
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007406
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038732, -0.0003915, -0.0039770, -0.0005928, -0.0028759, 0.0028932
1: -0.0040307, -0.0030490, -0.0040599, -0.0031058, -0.0008108, 0.0008157
2: 0.0088209, 0.0160635, 0.0086048, 0.0156447, -0.0059825, 0.0060184
3: 0.0027946, 0.0037530, 0.0027660, 0.0036976, -0.0007917, 0.0007964
4: -0.0059130, -0.0005003, -0.0056000, -0.0003388, -0.0044978, 0.0044710
5: 0.9938633, 0.9953673, 0.9939504, 0.9954121, -0.0012496, 0.0012422
6: 0.0023135, 0.0036785, 0.0023924, 0.0037192, -0.0011343, 0.0011275
7: -0.0147479, -0.0096540, -0.0144533, -0.0095020, -0.0042329, 0.0042077
8: -0.0016792, 0.0022855, -0.0017974, 0.0020562, -0.0032748, 0.0032945
9: -0.0042069, -0.0038649, -0.0041871, -0.0038547, -0.0002842, 0.0002825

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007168, upper bound: 0.0007410
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007168, upper bound: 0.0007415
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0037573, -0.0001683, -0.0039236, -0.0006211, -0.0027189, 0.0030467
1: -0.0039980, -0.0029861, -0.0040449, -0.0031138, -0.0007666, 0.0008590
2: 0.0090618, 0.0165277, 0.0087159, 0.0155859, -0.0056559, 0.0063377
3: 0.0028265, 0.0038145, 0.0027807, 0.0036898, -0.0007485, 0.0008387
4: -0.0062599, -0.0006804, -0.0055561, -0.0004218, -0.0047364, 0.0042269
5: 0.9937670, 0.9953172, 0.9939626, 0.9953890, -0.0013159, 0.0011744
6: 0.0022260, 0.0036331, 0.0024035, 0.0036983, -0.0011945, 0.0010660
7: -0.0150744, -0.0098235, -0.0144120, -0.0095801, -0.0044575, 0.0039780
8: -0.0015472, 0.0025396, -0.0017366, 0.0020240, -0.0030961, 0.0034693
9: -0.0042288, -0.0038762, -0.0041844, -0.0038599, -0.0002993, 0.0002671

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007204
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007212
time: 2.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037884, -0.0001520, -0.0039236, -0.0006227, -0.0027682, 0.0030606
1: -0.0040068, -0.0029815, -0.0040449, -0.0031142, -0.0007805, 0.0008629
2: 0.0089972, 0.0165617, 0.0087159, 0.0155825, -0.0057584, 0.0063666
3: 0.0028179, 0.0038190, 0.0027807, 0.0036894, -0.0007620, 0.0008425
4: -0.0062853, -0.0006321, -0.0055535, -0.0004218, -0.0047580, 0.0043035
5: 0.9937599, 0.9953306, 0.9939633, 0.9953891, -0.0013219, 0.0011956
6: 0.0022196, 0.0036453, 0.0024042, 0.0036983, -0.0011999, 0.0010853
7: -0.0150983, -0.0097780, -0.0144096, -0.0095801, -0.0044778, 0.0040500
8: -0.0015827, 0.0025582, -0.0017366, 0.0020222, -0.0031521, 0.0034851
9: -0.0042304, -0.0038732, -0.0041842, -0.0038599, -0.0003007, 0.0002720

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007208
time: 1.98 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007204
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0037576, -0.0001652, -0.0039771, -0.0005891, -0.0027463, 0.0031435
1: -0.0039981, -0.0029852, -0.0040599, -0.0031048, -0.0007743, 0.0008863
2: 0.0090612, 0.0165341, 0.0086047, 0.0156523, -0.0057129, 0.0065390
3: 0.0028264, 0.0038153, 0.0027660, 0.0036986, -0.0007560, 0.0008653
4: -0.0062647, -0.0006799, -0.0056057, -0.0003388, -0.0048869, 0.0042695
5: 0.9937658, 0.9953174, 0.9939489, 0.9954121, -0.0013577, 0.0011862
6: 0.0022248, 0.0036332, 0.0023910, 0.0037192, -0.0012324, 0.0010767
7: -0.0150789, -0.0098230, -0.0144587, -0.0095019, -0.0045991, 0.0040180
8: -0.0015476, 0.0025431, -0.0017975, 0.0020604, -0.0031272, 0.0035795
9: -0.0042291, -0.0038762, -0.0041875, -0.0038547, -0.0003088, 0.0002698

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007309
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007313
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037887, -0.0001494, -0.0039770, -0.0005928, -0.0027945, 0.0031572
1: -0.0040068, -0.0029808, -0.0040599, -0.0031058, -0.0007879, 0.0008901
2: 0.0089967, 0.0165671, 0.0086048, 0.0156447, -0.0058132, 0.0065676
3: 0.0028179, 0.0038197, 0.0027660, 0.0036976, -0.0007693, 0.0008691
4: -0.0062893, -0.0006317, -0.0056000, -0.0003388, -0.0049082, 0.0043444
5: 0.9937589, 0.9953306, 0.9939504, 0.9954121, -0.0013636, 0.0012070
6: 0.0022186, 0.0036454, 0.0023924, 0.0037192, -0.0012378, 0.0010956
7: -0.0151021, -0.0097776, -0.0144533, -0.0095020, -0.0046192, 0.0040886
8: -0.0015829, 0.0025611, -0.0017974, 0.0020562, -0.0031821, 0.0035951
9: -0.0042307, -0.0038732, -0.0041871, -0.0038547, -0.0003102, 0.0002745

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007309
time: 1.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007305
time: 2.13 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038632, -0.0004944, -0.0037639, -0.0002995, -0.0028988, 0.0026417
1: -0.0040278, -0.0030780, -0.0039999, -0.0030231, -0.0008173, 0.0007448
2: 0.0088417, 0.0158494, 0.0090481, 0.0162549, -0.0060301, 0.0054952
3: 0.0027974, 0.0037247, 0.0028247, 0.0037784, -0.0007980, 0.0007272
4: -0.0057530, -0.0005159, -0.0060560, -0.0006701, -0.0041068, 0.0045065
5: 0.9939078, 0.9953629, 0.9938236, 0.9953200, -0.0011410, 0.0012520
6: 0.0023538, 0.0036746, 0.0022774, 0.0036357, -0.0010357, 0.0011365
7: -0.0145973, -0.0096686, -0.0148825, -0.0098138, -0.0038649, 0.0042411
8: -0.0016678, 0.0021683, -0.0015548, 0.0023902, -0.0033009, 0.0030081
9: -0.0041968, -0.0038658, -0.0042160, -0.0038756, -0.0002595, 0.0002848

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007209
time: 1.92 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007280
time: 1.90 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0038934, -0.0004791, -0.0037638, -0.0003005, -0.0029330, 0.0026559
1: -0.0040364, -0.0030737, -0.0039998, -0.0030234, -0.0008269, 0.0007488
2: 0.0087787, 0.0158813, 0.0090483, 0.0162527, -0.0061012, 0.0055247
3: 0.0027890, 0.0037289, 0.0028247, 0.0037781, -0.0008074, 0.0007311
4: -0.0057768, -0.0004688, -0.0060544, -0.0006703, -0.0041288, 0.0045597
5: 0.9939013, 0.9953760, 0.9938241, 0.9953200, -0.0011471, 0.0012668
6: 0.0023478, 0.0036864, 0.0022778, 0.0036356, -0.0010412, 0.0011499
7: -0.0146197, -0.0096243, -0.0148810, -0.0098139, -0.0038857, 0.0042912
8: -0.0017022, 0.0021857, -0.0015547, 0.0023890, -0.0033398, 0.0030242
9: -0.0041983, -0.0038629, -0.0042159, -0.0038756, -0.0002609, 0.0002881

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007020, upper bound: 0.0007205
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007020, upper bound: 0.0007284
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038632, -0.0004944, -0.0037630, -0.0000855, -0.0030576, 0.0026089
1: -0.0040278, -0.0030780, -0.0039996, -0.0029628, -0.0008621, 0.0007355
2: 0.0088417, 0.0158494, 0.0090500, 0.0167000, -0.0063605, 0.0054270
3: 0.0027974, 0.0037247, 0.0028249, 0.0038373, -0.0008417, 0.0007182
4: -0.0057530, -0.0005159, -0.0063887, -0.0006715, -0.0040558, 0.0047534
5: 0.9939078, 0.9953629, 0.9937313, 0.9953196, -0.0011268, 0.0013206
6: 0.0023538, 0.0036746, 0.0021935, 0.0036353, -0.0010228, 0.0011987
7: -0.0145973, -0.0096686, -0.0151956, -0.0098151, -0.0038170, 0.0044735
8: -0.0016678, 0.0021683, -0.0015537, 0.0026339, -0.0034817, 0.0029707
9: -0.0041968, -0.0038658, -0.0042370, -0.0038757, -0.0002563, 0.0003004

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007205
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007280
time: 1.52 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038934, -0.0004791, -0.0037629, -0.0000865, -0.0030977, 0.0026241
1: -0.0040364, -0.0030737, -0.0039996, -0.0029630, -0.0008733, 0.0007398
2: 0.0087787, 0.0158813, 0.0090503, 0.0166980, -0.0064438, 0.0054586
3: 0.0027890, 0.0037289, 0.0028250, 0.0038370, -0.0008527, 0.0007224
4: -0.0057768, -0.0004688, -0.0063872, -0.0006717, -0.0040794, 0.0048157
5: 0.9939013, 0.9953760, 0.9937317, 0.9953196, -0.0011334, 0.0013379
6: 0.0023478, 0.0036864, 0.0021939, 0.0036353, -0.0010288, 0.0012144
7: -0.0146197, -0.0096243, -0.0151942, -0.0098153, -0.0038392, 0.0045321
8: -0.0017022, 0.0021857, -0.0015536, 0.0026328, -0.0035273, 0.0029880
9: -0.0041983, -0.0038629, -0.0042369, -0.0038757, -0.0002578, 0.0003043

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007016, upper bound: 0.0007205
time: 2.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007016, upper bound: 0.0007284
time: 1.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0039122, -0.0004646, -0.0037642, -0.0002960, -0.0029873, 0.0026691
1: -0.0040416, -0.0030696, -0.0039999, -0.0030221, -0.0008422, 0.0007525
2: 0.0087397, 0.0159114, 0.0090474, 0.0162620, -0.0062143, 0.0055524
3: 0.0027839, 0.0037329, 0.0028246, 0.0037793, -0.0008224, 0.0007348
4: -0.0057993, -0.0004397, -0.0060614, -0.0006696, -0.0041495, 0.0046442
5: 0.9938951, 0.9953841, 0.9938222, 0.9953203, -0.0011529, 0.0012903
6: 0.0023422, 0.0036938, 0.0022761, 0.0036358, -0.0010464, 0.0011712
7: -0.0146410, -0.0095969, -0.0148876, -0.0098133, -0.0039051, 0.0043707
8: -0.0017236, 0.0022022, -0.0015551, 0.0023941, -0.0034017, 0.0030394
9: -0.0041997, -0.0038610, -0.0042163, -0.0038756, -0.0002622, 0.0002935

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0007178
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007178
time: 1.98 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0039371, -0.0004552, -0.0037641, -0.0002973, -0.0030183, 0.0026819
1: -0.0040487, -0.0030670, -0.0039999, -0.0030225, -0.0008510, 0.0007561
2: 0.0086879, 0.0159310, 0.0090477, 0.0162594, -0.0062787, 0.0055790
3: 0.0027770, 0.0037355, 0.0028246, 0.0037790, -0.0008309, 0.0007383
4: -0.0058139, -0.0004009, -0.0060594, -0.0006698, -0.0041694, 0.0046923
5: 0.9938909, 0.9953949, 0.9938228, 0.9953201, -0.0011584, 0.0013037
6: 0.0023385, 0.0037036, 0.0022766, 0.0036358, -0.0010515, 0.0011833
7: -0.0146547, -0.0095604, -0.0148857, -0.0098135, -0.0039239, 0.0044160
8: -0.0017520, 0.0022129, -0.0015550, 0.0023927, -0.0034370, 0.0030540
9: -0.0042007, -0.0038586, -0.0042162, -0.0038756, -0.0002635, 0.0002965

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007186
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007183
time: 2.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0039122, -0.0004646, -0.0037633, -0.0000827, -0.0031600, 0.0026408
1: -0.0040416, -0.0030696, -0.0039997, -0.0029620, -0.0008909, 0.0007445
2: 0.0087397, 0.0159114, 0.0090494, 0.0167057, -0.0065734, 0.0054934
3: 0.0027839, 0.0037329, 0.0028248, 0.0038380, -0.0008699, 0.0007270
4: -0.0057993, -0.0004397, -0.0063930, -0.0006711, -0.0041055, 0.0049126
5: 0.9938951, 0.9953841, 0.9937301, 0.9953199, -0.0011406, 0.0013649
6: 0.0023422, 0.0036938, 0.0021925, 0.0036354, -0.0010353, 0.0012389
7: -0.0146410, -0.0095969, -0.0151996, -0.0098147, -0.0038637, 0.0046233
8: -0.0017236, 0.0022022, -0.0015541, 0.0026370, -0.0035983, 0.0030071
9: -0.0041997, -0.0038610, -0.0042372, -0.0038757, -0.0002594, 0.0003104

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006727, upper bound: 0.0006959
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006725, upper bound: 0.0006881
time: 1.98 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0039371, -0.0004552, -0.0037632, -0.0000838, -0.0031893, 0.0026536
1: -0.0040487, -0.0030670, -0.0039996, -0.0029623, -0.0008992, 0.0007482
2: 0.0086879, 0.0159310, 0.0090497, 0.0167036, -0.0066345, 0.0055201
3: 0.0027770, 0.0037355, 0.0028249, 0.0038377, -0.0008780, 0.0007305
4: -0.0058139, -0.0004009, -0.0063913, -0.0006713, -0.0041254, 0.0049582
5: 0.9938909, 0.9953949, 0.9937305, 0.9953197, -0.0011462, 0.0013775
6: 0.0023385, 0.0037036, 0.0021929, 0.0036354, -0.0010404, 0.0012504
7: -0.0146547, -0.0095604, -0.0151981, -0.0098149, -0.0038825, 0.0046662
8: -0.0017520, 0.0022129, -0.0015539, 0.0026358, -0.0036317, 0.0030217
9: -0.0042007, -0.0038586, -0.0042371, -0.0038757, -0.0002607, 0.0003133

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006751, upper bound: 0.0006958
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006889
time: 1.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.23 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007163, upper bound: 0.0007225
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007166, upper bound: 0.0007249
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007255, upper bound: 0.0007234
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007258, upper bound: 0.0007249
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007074
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007093
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007074
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007096
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007118
NS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007115
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007096, upper bound: 0.0007118
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007096, upper bound: 0.0007118
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007208
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007074, upper bound: 0.0007208
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007208
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007208
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007316, upper bound: 0.0007143
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007316, upper bound: 0.0007173
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007142
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007164
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007208, upper bound: 0.0006993
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007009
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007309, upper bound: 0.0006993
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007309, upper bound: 0.0007017
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007062, upper bound: 0.0007012
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007197, upper bound: 0.0007012
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007012
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007084, upper bound: 0.0007017
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007103
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007192, upper bound: 0.0007099
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007095
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007095
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007320
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007316
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0007312
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0007312
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007407
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007147, upper bound: 0.0007406
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007168, upper bound: 0.0007410
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007168, upper bound: 0.0007415
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007204
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007212
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007208
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007204
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007309
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007313
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007309
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007305
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007209
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007280
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007020, upper bound: 0.0007205
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007020, upper bound: 0.0007284
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007205
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006975, upper bound: 0.0007280
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007016, upper bound: 0.0007205
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007016, upper bound: 0.0007284
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007075, upper bound: 0.0007178
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007178
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007186
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007183
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006727, upper bound: 0.0006959
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006725, upper bound: 0.0006881
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006751, upper bound: 0.0006958
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006889

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006319, -0.0038428, -0.0006242, -0.0025827, 0.0027026
1: -0.0040221, -0.0031168, -0.0040221, -0.0031147, -0.0007282, 0.0007620
2: 0.0088841, 0.0155633, 0.0088841, 0.0155793, -0.0053726, 0.0056219
3: 0.0028030, 0.0036869, 0.0028030, 0.0036890, -0.0007110, 0.0007440
4: -0.0055392, -0.0005476, -0.0055512, -0.0005476, -0.0042015, 0.0040151
5: 0.9939673, 0.9953541, 0.9939640, 0.9953541, -0.0011673, 0.0011155
6: 0.0024078, 0.0036666, 0.0024047, 0.0036666, -0.0010596, 0.0010126
7: -0.0143961, -0.0096985, -0.0144074, -0.0096985, -0.0039541, 0.0037787
8: -0.0016445, 0.0020117, -0.0016445, 0.0020204, -0.0029410, 0.0030774
9: -0.0041833, -0.0038679, -0.0041840, -0.0038679, -0.0002655, 0.0002537

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006953, upper bound: 0.0006998
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006937, upper bound: 0.0006990
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006335, -0.0038708, -0.0006092, -0.0025971, 0.0027588
1: -0.0040221, -0.0031173, -0.0040300, -0.0031104, -0.0007322, 0.0007778
2: 0.0088840, 0.0155601, 0.0088257, 0.0156106, -0.0054024, 0.0057389
3: 0.0028030, 0.0036864, 0.0027952, 0.0036931, -0.0007149, 0.0007595
4: -0.0055368, -0.0005475, -0.0055746, -0.0005039, -0.0042889, 0.0040374
5: 0.9939680, 0.9953541, 0.9939574, 0.9953662, -0.0011916, 0.0011217
6: 0.0024084, 0.0036666, 0.0023988, 0.0036776, -0.0010816, 0.0010182
7: -0.0143939, -0.0096984, -0.0144294, -0.0096574, -0.0040363, 0.0037997
8: -0.0016446, 0.0020099, -0.0016765, 0.0020376, -0.0029573, 0.0031415
9: -0.0041831, -0.0038678, -0.0041855, -0.0038651, -0.0002710, 0.0002551

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006952, upper bound: 0.0007026
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006939, upper bound: 0.0007030
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006003, -0.0038434, -0.0006195, -0.0026962, 0.0027303
1: -0.0040377, -0.0031079, -0.0040222, -0.0031133, -0.0007601, 0.0007698
2: 0.0087692, 0.0156290, 0.0088829, 0.0155891, -0.0056086, 0.0056796
3: 0.0027878, 0.0036955, 0.0028028, 0.0036903, -0.0007422, 0.0007516
4: -0.0055883, -0.0004617, -0.0055585, -0.0005466, -0.0042446, 0.0041915
5: 0.9939536, 0.9953780, 0.9939619, 0.9953543, -0.0011793, 0.0011645
6: 0.0023954, 0.0036882, 0.0024029, 0.0036668, -0.0010704, 0.0010570
7: -0.0144423, -0.0096177, -0.0144143, -0.0096976, -0.0039946, 0.0039447
8: -0.0017074, 0.0020476, -0.0016452, 0.0020258, -0.0030701, 0.0031090
9: -0.0041864, -0.0038624, -0.0041845, -0.0038678, -0.0002682, 0.0002649

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007053, upper bound: 0.0006990
time: 1.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007027, upper bound: 0.0006990
time: 2.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006038, -0.0038714, -0.0006066, -0.0027103, 0.0026385
1: -0.0040376, -0.0031089, -0.0040302, -0.0031097, -0.0007641, 0.0007439
2: 0.0087693, 0.0156219, 0.0088245, 0.0156160, -0.0056380, 0.0054886
3: 0.0027878, 0.0036946, 0.0027951, 0.0036938, -0.0007461, 0.0007263
4: -0.0055830, -0.0004617, -0.0055785, -0.0005030, -0.0041018, 0.0042135
5: 0.9939551, 0.9953780, 0.9939563, 0.9953665, -0.0011396, 0.0011706
6: 0.0023967, 0.0036882, 0.0023978, 0.0036778, -0.0010344, 0.0010626
7: -0.0144373, -0.0096177, -0.0144332, -0.0096566, -0.0038603, 0.0039654
8: -0.0017074, 0.0020437, -0.0016772, 0.0020405, -0.0030863, 0.0030044
9: -0.0041861, -0.0038624, -0.0041858, -0.0038650, -0.0002592, 0.0002663

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007056, upper bound: 0.0007022
time: 1.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006935, upper bound: 0.0007021
time: 2.42 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006319, -0.0037582, -0.0003802, -0.0028528, 0.0026283
1: -0.0040221, -0.0031168, -0.0039982, -0.0030459, -0.0008043, 0.0007410
2: 0.0088841, 0.0155633, 0.0090600, 0.0160869, -0.0059343, 0.0054673
3: 0.0028030, 0.0036869, 0.0028262, 0.0037561, -0.0007853, 0.0007235
4: -0.0055392, -0.0005476, -0.0059305, -0.0006790, -0.0040859, 0.0044349
5: 0.9939673, 0.9953541, 0.9938586, 0.9953176, -0.0011352, 0.0012322
6: 0.0024078, 0.0036666, 0.0023091, 0.0036334, -0.0010304, 0.0011184
7: -0.0143961, -0.0096985, -0.0147644, -0.0098222, -0.0038453, 0.0041738
8: -0.0016445, 0.0020117, -0.0015483, 0.0022983, -0.0032485, 0.0029928
9: -0.0041833, -0.0038679, -0.0042080, -0.0038762, -0.0002582, 0.0002803

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006836, upper bound: 0.0006771
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006803, upper bound: 0.0006768
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006335, -0.0037889, -0.0003642, -0.0028676, 0.0026785
1: -0.0040221, -0.0031173, -0.0040069, -0.0030413, -0.0008085, 0.0007552
2: 0.0088840, 0.0155601, 0.0089961, 0.0161203, -0.0059651, 0.0055719
3: 0.0028030, 0.0036864, 0.0028178, 0.0037606, -0.0007894, 0.0007374
4: -0.0055368, -0.0005475, -0.0059554, -0.0006313, -0.0041641, 0.0044580
5: 0.9939680, 0.9953541, 0.9938516, 0.9953309, -0.0011569, 0.0012386
6: 0.0024084, 0.0036666, 0.0023028, 0.0036455, -0.0010501, 0.0011242
7: -0.0143939, -0.0096984, -0.0147879, -0.0097772, -0.0039189, 0.0041954
8: -0.0016446, 0.0020099, -0.0015832, 0.0023166, -0.0032653, 0.0030501
9: -0.0041831, -0.0038678, -0.0042096, -0.0038731, -0.0002631, 0.0002817

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006840, upper bound: 0.0006795
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006782, upper bound: 0.0006795
time: 2.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006003, -0.0037586, -0.0003767, -0.0029628, 0.0026556
1: -0.0040377, -0.0031079, -0.0039983, -0.0030449, -0.0008353, 0.0007487
2: 0.0087692, 0.0156290, 0.0090593, 0.0160943, -0.0061632, 0.0055243
3: 0.0027878, 0.0036955, 0.0028261, 0.0037571, -0.0008156, 0.0007311
4: -0.0055883, -0.0004617, -0.0059360, -0.0006785, -0.0041285, 0.0046060
5: 0.9939536, 0.9953780, 0.9938570, 0.9953178, -0.0011470, 0.0012797
6: 0.0023954, 0.0036882, 0.0023077, 0.0036336, -0.0010411, 0.0011616
7: -0.0144423, -0.0096177, -0.0147696, -0.0098217, -0.0038854, 0.0043347
8: -0.0017074, 0.0020476, -0.0015486, 0.0023023, -0.0033737, 0.0030240
9: -0.0041864, -0.0038624, -0.0042084, -0.0038761, -0.0002609, 0.0002911

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006931, upper bound: 0.0006771
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006889, upper bound: 0.0006771
time: 2.15 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006038, -0.0037892, -0.0003615, -0.0029773, 0.0025697
1: -0.0040376, -0.0031089, -0.0040070, -0.0030406, -0.0008394, 0.0007245
2: 0.0087693, 0.0156219, 0.0089955, 0.0161258, -0.0061934, 0.0053455
3: 0.0027878, 0.0036946, 0.0028177, 0.0037613, -0.0008196, 0.0007074
4: -0.0055830, -0.0004617, -0.0059596, -0.0006308, -0.0039949, 0.0046286
5: 0.9939551, 0.9953780, 0.9938505, 0.9953310, -0.0011099, 0.0012860
6: 0.0023967, 0.0036882, 0.0023018, 0.0036456, -0.0010075, 0.0011673
7: -0.0144373, -0.0096177, -0.0147918, -0.0097768, -0.0037596, 0.0043560
8: -0.0017074, 0.0020437, -0.0015836, 0.0023196, -0.0033903, 0.0029261
9: -0.0041861, -0.0038624, -0.0042099, -0.0038731, -0.0002525, 0.0002925

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006931, upper bound: 0.0006791
time: 1.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006889, upper bound: 0.0006794
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0037582, -0.0003802, -0.0038428, -0.0006319, -0.0026283, 0.0028528
1: -0.0039982, -0.0030459, -0.0040221, -0.0031168, -0.0007410, 0.0008043
2: 0.0090600, 0.0160869, 0.0088841, 0.0155633, -0.0054673, 0.0059343
3: 0.0028262, 0.0037561, 0.0028030, 0.0036869, -0.0007235, 0.0007853
4: -0.0059305, -0.0006790, -0.0055392, -0.0005476, -0.0044349, 0.0040859
5: 0.9938586, 0.9953176, 0.9939673, 0.9953541, -0.0012322, 0.0011352
6: 0.0023091, 0.0036334, 0.0024078, 0.0036666, -0.0011184, 0.0010304
7: -0.0147644, -0.0098222, -0.0143961, -0.0096985, -0.0041738, 0.0038453
8: -0.0015483, 0.0022983, -0.0016445, 0.0020117, -0.0029928, 0.0032485
9: -0.0042080, -0.0038762, -0.0041833, -0.0038679, -0.0002803, 0.0002582

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006995, upper bound: 0.0007119
time: 1.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006995, upper bound: 0.0007115
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037889, -0.0003642, -0.0038428, -0.0006335, -0.0026785, 0.0028676
1: -0.0040069, -0.0030413, -0.0040221, -0.0031173, -0.0007552, 0.0008085
2: 0.0089961, 0.0161203, 0.0088840, 0.0155601, -0.0055719, 0.0059651
3: 0.0028178, 0.0037606, 0.0028030, 0.0036864, -0.0007374, 0.0007894
4: -0.0059554, -0.0006313, -0.0055368, -0.0005475, -0.0044580, 0.0041641
5: 0.9938516, 0.9953309, 0.9939680, 0.9953541, -0.0012386, 0.0011569
6: 0.0023028, 0.0036455, 0.0024084, 0.0036666, -0.0011242, 0.0010501
7: -0.0147879, -0.0097772, -0.0143939, -0.0096984, -0.0041954, 0.0039189
8: -0.0015832, 0.0023166, -0.0016446, 0.0020099, -0.0030501, 0.0032653
9: -0.0042096, -0.0038731, -0.0041831, -0.0038678, -0.0002817, 0.0002631

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007023, upper bound: 0.0007118
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007023, upper bound: 0.0007115
time: 2.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037889, -0.0003642, -0.0037596, -0.0003907, -0.0026787, 0.0026760
1: -0.0040069, -0.0030413, -0.0039986, -0.0030488, -0.0007552, 0.0007545
2: 0.0089961, 0.0161203, 0.0090570, 0.0160651, -0.0055722, 0.0055667
3: 0.0028178, 0.0037606, 0.0028259, 0.0037533, -0.0007374, 0.0007367
4: -0.0059554, -0.0006313, -0.0059142, -0.0006768, -0.0041602, 0.0041643
5: 0.9938516, 0.9953309, 0.9938631, 0.9953182, -0.0011558, 0.0011570
6: 0.0023028, 0.0036455, 0.0023132, 0.0036340, -0.0010491, 0.0010502
7: -0.0147879, -0.0097772, -0.0147490, -0.0098201, -0.0039152, 0.0039191
8: -0.0015832, 0.0023166, -0.0015499, 0.0022863, -0.0030502, 0.0030472
9: -0.0042096, -0.0038731, -0.0042070, -0.0038760, -0.0002629, 0.0002632

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007023, upper bound: 0.0007118
time: 2.42 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007023, upper bound: 0.0007118
time: 1.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0037586, -0.0003767, -0.0038980, -0.0006003, -0.0026556, 0.0029628
1: -0.0039983, -0.0030449, -0.0040377, -0.0031079, -0.0007487, 0.0008353
2: 0.0090593, 0.0160943, 0.0087692, 0.0156290, -0.0055243, 0.0061632
3: 0.0028261, 0.0037571, 0.0027878, 0.0036955, -0.0007311, 0.0008156
4: -0.0059360, -0.0006785, -0.0055883, -0.0004617, -0.0046060, 0.0041285
5: 0.9938570, 0.9953178, 0.9939536, 0.9953780, -0.0012797, 0.0011470
6: 0.0023077, 0.0036336, 0.0023954, 0.0036882, -0.0011616, 0.0010411
7: -0.0147696, -0.0098217, -0.0144423, -0.0096177, -0.0043347, 0.0038854
8: -0.0015486, 0.0023023, -0.0017074, 0.0020476, -0.0030240, 0.0033737
9: -0.0042084, -0.0038761, -0.0041864, -0.0038624, -0.0002911, 0.0002609

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006963, upper bound: 0.0007212
time: 2.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006963, upper bound: 0.0007122
time: 2.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0037586, -0.0003767, -0.0038102, -0.0003593, -0.0026680, 0.0027680
1: -0.0039983, -0.0030449, -0.0040129, -0.0030400, -0.0007522, 0.0007804
2: 0.0090593, 0.0160943, 0.0089518, 0.0161304, -0.0055500, 0.0057580
3: 0.0028261, 0.0037571, 0.0028119, 0.0037619, -0.0007345, 0.0007620
4: -0.0059360, -0.0006785, -0.0059630, -0.0005981, -0.0043032, 0.0041478
5: 0.9938570, 0.9953178, 0.9938495, 0.9953401, -0.0011955, 0.0011524
6: 0.0023077, 0.0036336, 0.0023009, 0.0036538, -0.0010852, 0.0010460
7: -0.0147696, -0.0098217, -0.0147950, -0.0097461, -0.0040498, 0.0039035
8: -0.0015486, 0.0023023, -0.0016075, 0.0023221, -0.0030381, 0.0031519
9: -0.0042084, -0.0038761, -0.0042101, -0.0038710, -0.0002719, 0.0002621

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006963, upper bound: 0.0007212
time: 2.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006963, upper bound: 0.0007118
time: 2.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037892, -0.0003615, -0.0038980, -0.0006038, -0.0025697, 0.0029773
1: -0.0040070, -0.0030406, -0.0040376, -0.0031089, -0.0007245, 0.0008394
2: 0.0089955, 0.0161258, 0.0087693, 0.0156219, -0.0053455, 0.0061934
3: 0.0028177, 0.0037613, 0.0027878, 0.0036946, -0.0007074, 0.0008196
4: -0.0059596, -0.0006308, -0.0055830, -0.0004617, -0.0046286, 0.0039949
5: 0.9938505, 0.9953310, 0.9939551, 0.9953780, -0.0012860, 0.0011099
6: 0.0023018, 0.0036456, 0.0023967, 0.0036882, -0.0011673, 0.0010075
7: -0.0147918, -0.0097768, -0.0144373, -0.0096177, -0.0043560, 0.0037596
8: -0.0015836, 0.0023196, -0.0017074, 0.0020437, -0.0029261, 0.0033903
9: -0.0042099, -0.0038731, -0.0041861, -0.0038624, -0.0002925, 0.0002525

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007000, upper bound: 0.0007216
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007000, upper bound: 0.0007118
time: 1.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037892, -0.0003615, -0.0038101, -0.0003609, -0.0027086, 0.0027829
1: -0.0040070, -0.0030406, -0.0040129, -0.0030404, -0.0007637, 0.0007846
2: 0.0089955, 0.0161258, 0.0089520, 0.0161270, -0.0056345, 0.0057890
3: 0.0028177, 0.0037613, 0.0028120, 0.0037615, -0.0007456, 0.0007661
4: -0.0059596, -0.0006308, -0.0059605, -0.0005983, -0.0043264, 0.0042109
5: 0.9938505, 0.9953310, 0.9938502, 0.9953400, -0.0012020, 0.0011699
6: 0.0023018, 0.0036456, 0.0023015, 0.0036538, -0.0010910, 0.0010619
7: -0.0147918, -0.0097768, -0.0147926, -0.0097462, -0.0040716, 0.0039629
8: -0.0015836, 0.0023196, -0.0016074, 0.0023203, -0.0030843, 0.0031689
9: -0.0042099, -0.0038731, -0.0042099, -0.0038711, -0.0002734, 0.0002661

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007000, upper bound: 0.0007208
time: 1.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007000, upper bound: 0.0007118
time: 2.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006319, -0.0038425, -0.0004058, -0.0027305, 0.0026471
1: -0.0040221, -0.0031168, -0.0040220, -0.0030531, -0.0007698, 0.0007463
2: 0.0088841, 0.0155633, 0.0088847, 0.0160338, -0.0056800, 0.0055066
3: 0.0028030, 0.0036869, 0.0028030, 0.0037491, -0.0007517, 0.0007287
4: -0.0055392, -0.0005476, -0.0058908, -0.0005480, -0.0041153, 0.0042449
5: 0.9939673, 0.9953541, 0.9938695, 0.9953540, -0.0011433, 0.0011794
6: 0.0024078, 0.0036666, 0.0023191, 0.0036665, -0.0010378, 0.0010705
7: -0.0143961, -0.0096985, -0.0147270, -0.0096989, -0.0038729, 0.0039949
8: -0.0016445, 0.0020117, -0.0016442, 0.0022692, -0.0031093, 0.0030143
9: -0.0041833, -0.0038679, -0.0042055, -0.0038679, -0.0002601, 0.0002683

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0006920
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0006920
time: 2.15 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006335, -0.0038727, -0.0003929, -0.0027441, 0.0027021
1: -0.0040221, -0.0031173, -0.0040305, -0.0030494, -0.0007737, 0.0007618
2: 0.0088840, 0.0155601, 0.0088218, 0.0160606, -0.0057082, 0.0056208
3: 0.0028030, 0.0036864, 0.0027947, 0.0037527, -0.0007554, 0.0007438
4: -0.0055368, -0.0005475, -0.0059108, -0.0005010, -0.0042007, 0.0042660
5: 0.9939680, 0.9953541, 0.9938640, 0.9953670, -0.0011671, 0.0011852
6: 0.0024084, 0.0036666, 0.0023140, 0.0036783, -0.0010593, 0.0010758
7: -0.0143939, -0.0096984, -0.0147459, -0.0096546, -0.0039533, 0.0040148
8: -0.0016446, 0.0020099, -0.0016787, 0.0022839, -0.0031247, 0.0030769
9: -0.0041831, -0.0038678, -0.0042068, -0.0038649, -0.0002655, 0.0002696

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007107, upper bound: 0.0006948
time: 2.20 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0006948
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006003, -0.0038430, -0.0004021, -0.0028407, 0.0026747
1: -0.0040377, -0.0031079, -0.0040221, -0.0030520, -0.0008009, 0.0007541
2: 0.0087692, 0.0156290, 0.0088837, 0.0160414, -0.0059092, 0.0055639
3: 0.0027878, 0.0036955, 0.0028029, 0.0037501, -0.0007820, 0.0007363
4: -0.0055883, -0.0004617, -0.0058965, -0.0005473, -0.0041581, 0.0044161
5: 0.9939536, 0.9953780, 0.9938680, 0.9953542, -0.0011552, 0.0012269
6: 0.0023954, 0.0036882, 0.0023177, 0.0036667, -0.0010486, 0.0011137
7: -0.0144423, -0.0096177, -0.0147324, -0.0096982, -0.0039132, 0.0041561
8: -0.0017074, 0.0020476, -0.0016448, 0.0022734, -0.0032347, 0.0030457
9: -0.0041864, -0.0038624, -0.0042059, -0.0038678, -0.0002628, 0.0002791

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007205, upper bound: 0.0006924
time: 2.42 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0006919
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006038, -0.0038732, -0.0003915, -0.0028539, 0.0025925
1: -0.0040376, -0.0031089, -0.0040307, -0.0030490, -0.0008046, 0.0007309
2: 0.0087693, 0.0156219, 0.0088209, 0.0160635, -0.0059367, 0.0053929
3: 0.0027878, 0.0036946, 0.0027946, 0.0037530, -0.0007856, 0.0007137
4: -0.0055830, -0.0004617, -0.0059130, -0.0005003, -0.0040303, 0.0044367
5: 0.9939551, 0.9953780, 0.9938633, 0.9953673, -0.0011198, 0.0012326
6: 0.0023967, 0.0036882, 0.0023135, 0.0036785, -0.0010164, 0.0011189
7: -0.0144373, -0.0096177, -0.0147479, -0.0096540, -0.0037930, 0.0041754
8: -0.0017074, 0.0020437, -0.0016792, 0.0022855, -0.0032497, 0.0029521
9: -0.0041861, -0.0038624, -0.0042069, -0.0038649, -0.0002547, 0.0002804

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007205, upper bound: 0.0006948
time: 2.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007177, upper bound: 0.0006948
time: 2.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006319, -0.0037573, -0.0001683, -0.0029993, 0.0025748
1: -0.0040221, -0.0031168, -0.0039980, -0.0029861, -0.0008456, 0.0007259
2: 0.0088841, 0.0155633, 0.0090618, 0.0165277, -0.0062391, 0.0053560
3: 0.0028030, 0.0036869, 0.0028265, 0.0038145, -0.0008256, 0.0007088
4: -0.0055392, -0.0005476, -0.0062599, -0.0006804, -0.0040028, 0.0046627
5: 0.9939673, 0.9953541, 0.9937670, 0.9953172, -0.0011121, 0.0012954
6: 0.0024078, 0.0036666, 0.0022260, 0.0036331, -0.0010094, 0.0011759
7: -0.0143961, -0.0096985, -0.0150744, -0.0098235, -0.0037671, 0.0043881
8: -0.0016445, 0.0020117, -0.0015472, 0.0025396, -0.0034153, 0.0029319
9: -0.0041833, -0.0038679, -0.0042288, -0.0038762, -0.0002530, 0.0002947

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006939, upper bound: 0.0006688
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006783, upper bound: 0.0006688
time: 2.44 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006335, -0.0037884, -0.0001520, -0.0030132, 0.0026237
1: -0.0040221, -0.0031173, -0.0040068, -0.0029815, -0.0008495, 0.0007397
2: 0.0088840, 0.0155601, 0.0089972, 0.0165617, -0.0062682, 0.0054578
3: 0.0028030, 0.0036864, 0.0028179, 0.0038190, -0.0008295, 0.0007223
4: -0.0055368, -0.0005475, -0.0062853, -0.0006321, -0.0040789, 0.0046844
5: 0.9939680, 0.9953541, 0.9937599, 0.9953306, -0.0011332, 0.0013015
6: 0.0024084, 0.0036666, 0.0022196, 0.0036453, -0.0010286, 0.0011813
7: -0.0143939, -0.0096984, -0.0150983, -0.0097780, -0.0038387, 0.0044086
8: -0.0016446, 0.0020099, -0.0015827, 0.0025582, -0.0034312, 0.0029876
9: -0.0041831, -0.0038678, -0.0042304, -0.0038732, -0.0002578, 0.0002960

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006939, upper bound: 0.0006707
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006886, upper bound: 0.0006703
time: 2.05 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006003, -0.0037576, -0.0001652, -0.0031041, 0.0026021
1: -0.0040377, -0.0031079, -0.0039981, -0.0029852, -0.0008752, 0.0007336
2: 0.0087692, 0.0156290, 0.0090612, 0.0165341, -0.0064572, 0.0054129
3: 0.0027878, 0.0036955, 0.0028264, 0.0038153, -0.0008545, 0.0007163
4: -0.0055883, -0.0004617, -0.0062647, -0.0006799, -0.0040452, 0.0048257
5: 0.9939536, 0.9953780, 0.9937658, 0.9953174, -0.0011239, 0.0013407
6: 0.0023954, 0.0036882, 0.0022248, 0.0036332, -0.0010202, 0.0012170
7: -0.0144423, -0.0096177, -0.0150789, -0.0098230, -0.0038070, 0.0045416
8: -0.0017074, 0.0020476, -0.0015476, 0.0025431, -0.0035347, 0.0029630
9: -0.0041864, -0.0038624, -0.0042291, -0.0038762, -0.0002556, 0.0003050

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007035, upper bound: 0.0006692
time: 2.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006688
time: 1.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038980, -0.0006038, -0.0037887, -0.0001494, -0.0031179, 0.0025218
1: -0.0040376, -0.0031089, -0.0040068, -0.0029808, -0.0008791, 0.0007110
2: 0.0087693, 0.0156219, 0.0089967, 0.0165671, -0.0064859, 0.0052458
3: 0.0027878, 0.0036946, 0.0028179, 0.0038197, -0.0008583, 0.0006942
4: -0.0055830, -0.0004617, -0.0062893, -0.0006317, -0.0039204, 0.0048471
5: 0.9939551, 0.9953780, 0.9937589, 0.9953306, -0.0010892, 0.0013467
6: 0.0023967, 0.0036882, 0.0022186, 0.0036454, -0.0009887, 0.0012224
7: -0.0144373, -0.0096177, -0.0151021, -0.0097776, -0.0036895, 0.0045617
8: -0.0017074, 0.0020437, -0.0015829, 0.0025611, -0.0035504, 0.0028716
9: -0.0041861, -0.0038624, -0.0042307, -0.0038732, -0.0002477, 0.0003063

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007035, upper bound: 0.0006704
time: 2.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006703
time: 2.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0037582, -0.0003802, -0.0037592, -0.0001766, -0.0027854, 0.0026150
1: -0.0039982, -0.0030459, -0.0039985, -0.0029884, -0.0007853, 0.0007373
2: 0.0090600, 0.0160869, 0.0090580, 0.0165105, -0.0057942, 0.0054397
3: 0.0028262, 0.0037561, 0.0028260, 0.0038122, -0.0007668, 0.0007199
4: -0.0059305, -0.0006790, -0.0062470, -0.0006775, -0.0040653, 0.0043302
5: 0.9938586, 0.9953176, 0.9937707, 0.9953180, -0.0011295, 0.0012031
6: 0.0023091, 0.0036334, 0.0022293, 0.0036338, -0.0010252, 0.0010920
7: -0.0147644, -0.0098222, -0.0150623, -0.0098207, -0.0038259, 0.0040752
8: -0.0015483, 0.0022983, -0.0015494, 0.0025301, -0.0031717, 0.0029777
9: -0.0042080, -0.0038762, -0.0042280, -0.0038761, -0.0002569, 0.0002736

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006979, upper bound: 0.0007017
time: 2.40 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006979, upper bound: 0.0007016
time: 2.38 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037889, -0.0003642, -0.0038431, -0.0004156, -0.0026882, 0.0028224
1: -0.0040069, -0.0030413, -0.0040222, -0.0030558, -0.0007579, 0.0007957
2: 0.0089961, 0.0161203, 0.0088834, 0.0160133, -0.0055921, 0.0058711
3: 0.0028178, 0.0037606, 0.0028029, 0.0037464, -0.0007400, 0.0007769
4: -0.0059554, -0.0006313, -0.0058755, -0.0005470, -0.0043877, 0.0041792
5: 0.9938516, 0.9953309, 0.9938738, 0.9953542, -0.0012190, 0.0011611
6: 0.0023028, 0.0036455, 0.0023230, 0.0036667, -0.0011065, 0.0010539
7: -0.0147879, -0.0097772, -0.0147126, -0.0096980, -0.0041293, 0.0039330
8: -0.0015832, 0.0023166, -0.0016449, 0.0022580, -0.0030611, 0.0032138
9: -0.0042096, -0.0038731, -0.0042045, -0.0038678, -0.0002773, 0.0002641

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007138, upper bound: 0.0007020
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007138, upper bound: 0.0007012
time: 2.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0037586, -0.0003767, -0.0038939, -0.0003862, -0.0026788, 0.0029033
1: -0.0039983, -0.0030449, -0.0040365, -0.0030476, -0.0007552, 0.0008186
2: 0.0090593, 0.0160943, 0.0087777, 0.0160744, -0.0055724, 0.0060395
3: 0.0028261, 0.0037571, 0.0027889, 0.0037545, -0.0007374, 0.0007992
4: -0.0059360, -0.0006785, -0.0059211, -0.0004680, -0.0045135, 0.0041645
5: 0.9938570, 0.9953178, 0.9938611, 0.9953762, -0.0012540, 0.0011570
6: 0.0023077, 0.0036336, 0.0023114, 0.0036866, -0.0011383, 0.0010502
7: -0.0147696, -0.0098217, -0.0147556, -0.0096236, -0.0042478, 0.0039192
8: -0.0015486, 0.0023023, -0.0017028, 0.0022914, -0.0030503, 0.0033060
9: -0.0042084, -0.0038761, -0.0042074, -0.0038628, -0.0002852, 0.0002632

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007090, upper bound: 0.0007095
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007090, upper bound: 0.0007012
time: 2.37 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0037586, -0.0003767, -0.0038060, -0.0001478, -0.0028135, 0.0027071
1: -0.0039983, -0.0030449, -0.0040117, -0.0029803, -0.0007932, 0.0007632
2: 0.0090593, 0.0160943, 0.0089606, 0.0165704, -0.0058527, 0.0056313
3: 0.0028261, 0.0037571, 0.0028131, 0.0038201, -0.0007745, 0.0007452
4: -0.0059360, -0.0006785, -0.0062918, -0.0006047, -0.0042085, 0.0043739
5: 0.9938570, 0.9953178, 0.9937582, 0.9953383, -0.0011692, 0.0012152
6: 0.0023077, 0.0036336, 0.0022180, 0.0036522, -0.0010613, 0.0011030
7: -0.0147696, -0.0098217, -0.0151044, -0.0097522, -0.0039607, 0.0041163
8: -0.0015486, 0.0023023, -0.0016027, 0.0025629, -0.0032038, 0.0030826
9: -0.0042084, -0.0038761, -0.0042309, -0.0038715, -0.0002660, 0.0002764

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0007099
time: 2.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007085, upper bound: 0.0007012
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037892, -0.0003615, -0.0038938, -0.0003890, -0.0027155, 0.0029178
1: -0.0040070, -0.0030406, -0.0040365, -0.0030483, -0.0007656, 0.0008226
2: 0.0089955, 0.0161258, 0.0087779, 0.0160687, -0.0056487, 0.0060697
3: 0.0028177, 0.0037613, 0.0027889, 0.0037537, -0.0007475, 0.0008032
4: -0.0059596, -0.0006308, -0.0059169, -0.0004682, -0.0045361, 0.0042215
5: 0.9938505, 0.9953310, 0.9938624, 0.9953762, -0.0012603, 0.0011729
6: 0.0023018, 0.0036456, 0.0023125, 0.0036866, -0.0011439, 0.0010646
7: -0.0147918, -0.0097768, -0.0147516, -0.0096237, -0.0042690, 0.0039729
8: -0.0015836, 0.0023196, -0.0017027, 0.0022883, -0.0030921, 0.0033225
9: -0.0042099, -0.0038731, -0.0042072, -0.0038628, -0.0002867, 0.0002668

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007117, upper bound: 0.0007099
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007117, upper bound: 0.0007016
time: 2.37 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037892, -0.0003615, -0.0038058, -0.0001488, -0.0028521, 0.0027220
1: -0.0040070, -0.0030406, -0.0040117, -0.0029806, -0.0008041, 0.0007674
2: 0.0089955, 0.0161258, 0.0089609, 0.0165684, -0.0059329, 0.0056623
3: 0.0028177, 0.0037613, 0.0028131, 0.0038199, -0.0007851, 0.0007493
4: -0.0059596, -0.0006308, -0.0062903, -0.0006050, -0.0042316, 0.0044339
5: 0.9938505, 0.9953310, 0.9937586, 0.9953382, -0.0011757, 0.0012319
6: 0.0023018, 0.0036456, 0.0022183, 0.0036521, -0.0010672, 0.0011182
7: -0.0147918, -0.0097768, -0.0151030, -0.0097525, -0.0039824, 0.0041728
8: -0.0015836, 0.0023196, -0.0016025, 0.0025619, -0.0032477, 0.0030995
9: -0.0042099, -0.0038731, -0.0042308, -0.0038715, -0.0002674, 0.0002802

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007117, upper bound: 0.0007099
time: 2.19 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007117, upper bound: 0.0007012
time: 2.15 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038425, -0.0004058, -0.0038428, -0.0006319, -0.0026471, 0.0027305
1: -0.0040220, -0.0030531, -0.0040221, -0.0031168, -0.0007463, 0.0007698
2: 0.0088847, 0.0160338, 0.0088841, 0.0155633, -0.0055066, 0.0056800
3: 0.0028030, 0.0037491, 0.0028030, 0.0036869, -0.0007287, 0.0007517
4: -0.0058908, -0.0005480, -0.0055392, -0.0005476, -0.0042449, 0.0041153
5: 0.9938695, 0.9953540, 0.9939673, 0.9953541, -0.0011794, 0.0011433
6: 0.0023191, 0.0036665, 0.0024078, 0.0036666, -0.0010705, 0.0010378
7: -0.0147270, -0.0096989, -0.0143961, -0.0096985, -0.0039949, 0.0038729
8: -0.0016442, 0.0022692, -0.0016445, 0.0020117, -0.0030143, 0.0031093
9: -0.0042055, -0.0038679, -0.0041833, -0.0038679, -0.0002683, 0.0002601

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007068, upper bound: 0.0007316
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007068, upper bound: 0.0007312
time: 2.23 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038425, -0.0004058, -0.0038431, -0.0004145, -0.0026832, 0.0027026
1: -0.0040220, -0.0030531, -0.0040222, -0.0030555, -0.0007565, 0.0007620
2: 0.0088847, 0.0160338, 0.0088834, 0.0160156, -0.0055817, 0.0056219
3: 0.0028030, 0.0037491, 0.0028029, 0.0037467, -0.0007386, 0.0007440
4: -0.0058908, -0.0005480, -0.0058772, -0.0005470, -0.0042014, 0.0041714
5: 0.9938695, 0.9953540, 0.9938735, 0.9953542, -0.0011673, 0.0011589
6: 0.0023191, 0.0036665, 0.0023225, 0.0036667, -0.0010595, 0.0010520
7: -0.0147270, -0.0096989, -0.0147142, -0.0096980, -0.0039540, 0.0039258
8: -0.0016442, 0.0022692, -0.0016449, 0.0022592, -0.0030554, 0.0030774
9: -0.0042055, -0.0038679, -0.0042047, -0.0038678, -0.0002655, 0.0002636

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007068, upper bound: 0.0007316
time: 2.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007068, upper bound: 0.0007312
time: 2.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0038727, -0.0003929, -0.0038428, -0.0006335, -0.0027021, 0.0027441
1: -0.0040305, -0.0030494, -0.0040221, -0.0031173, -0.0007618, 0.0007737
2: 0.0088218, 0.0160606, 0.0088840, 0.0155601, -0.0056208, 0.0057082
3: 0.0027947, 0.0037527, 0.0028030, 0.0036864, -0.0007438, 0.0007554
4: -0.0059108, -0.0005010, -0.0055368, -0.0005475, -0.0042660, 0.0042007
5: 0.9938640, 0.9953670, 0.9939680, 0.9953541, -0.0011852, 0.0011671
6: 0.0023140, 0.0036783, 0.0024084, 0.0036666, -0.0010758, 0.0010593
7: -0.0147459, -0.0096546, -0.0143939, -0.0096984, -0.0040148, 0.0039533
8: -0.0016787, 0.0022839, -0.0016446, 0.0020099, -0.0030769, 0.0031247
9: -0.0042068, -0.0038649, -0.0041831, -0.0038678, -0.0002696, 0.0002655

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007105, upper bound: 0.0007316
time: 2.03 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007105, upper bound: 0.0007316
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038727, -0.0003929, -0.0038431, -0.0004156, -0.0027280, 0.0027168
1: -0.0040305, -0.0030494, -0.0040222, -0.0030558, -0.0007691, 0.0007660
2: 0.0088218, 0.0160606, 0.0088834, 0.0160133, -0.0056747, 0.0056515
3: 0.0027947, 0.0037527, 0.0028029, 0.0037464, -0.0007510, 0.0007479
4: -0.0059108, -0.0005010, -0.0058755, -0.0005470, -0.0042236, 0.0042409
5: 0.9938640, 0.9953670, 0.9938738, 0.9953542, -0.0011734, 0.0011783
6: 0.0023140, 0.0036783, 0.0023230, 0.0036667, -0.0010651, 0.0010695
7: -0.0147459, -0.0096546, -0.0147126, -0.0096980, -0.0039749, 0.0039912
8: -0.0016787, 0.0022839, -0.0016449, 0.0022580, -0.0031064, 0.0030937
9: -0.0042068, -0.0038649, -0.0042045, -0.0038678, -0.0002669, 0.0002680

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007109, upper bound: 0.0007312
time: 2.27 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007109, upper bound: 0.0007316
time: 2.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038430, -0.0004021, -0.0038980, -0.0006003, -0.0026747, 0.0028407
1: -0.0040221, -0.0030520, -0.0040377, -0.0031079, -0.0007541, 0.0008009
2: 0.0088837, 0.0160414, 0.0087692, 0.0156290, -0.0055639, 0.0059092
3: 0.0028029, 0.0037501, 0.0027878, 0.0036955, -0.0007363, 0.0007820
4: -0.0058965, -0.0005473, -0.0055883, -0.0004617, -0.0044161, 0.0041581
5: 0.9938680, 0.9953542, 0.9939536, 0.9953780, -0.0012269, 0.0011552
6: 0.0023177, 0.0036667, 0.0023954, 0.0036882, -0.0011137, 0.0010486
7: -0.0147324, -0.0096982, -0.0144423, -0.0096177, -0.0041561, 0.0039132
8: -0.0016448, 0.0022734, -0.0017074, 0.0020476, -0.0030457, 0.0032347
9: -0.0042059, -0.0038678, -0.0041864, -0.0038624, -0.0002791, 0.0002628

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006935, upper bound: 0.0007173
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006924, upper bound: 0.0007169
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038430, -0.0004021, -0.0038939, -0.0003862, -0.0027137, 0.0028138
1: -0.0040221, -0.0030520, -0.0040365, -0.0030476, -0.0007651, 0.0007933
2: 0.0088837, 0.0160414, 0.0087777, 0.0160744, -0.0056451, 0.0058532
3: 0.0028029, 0.0037501, 0.0027889, 0.0037545, -0.0007470, 0.0007746
4: -0.0058965, -0.0005473, -0.0059211, -0.0004680, -0.0043743, 0.0042188
5: 0.9938680, 0.9953542, 0.9938611, 0.9953762, -0.0012153, 0.0011721
6: 0.0023177, 0.0036667, 0.0023114, 0.0036866, -0.0011031, 0.0010639
7: -0.0147324, -0.0096982, -0.0147556, -0.0096236, -0.0041167, 0.0039703
8: -0.0016448, 0.0022734, -0.0017028, 0.0022914, -0.0030901, 0.0032041
9: -0.0042059, -0.0038678, -0.0042074, -0.0038628, -0.0002764, 0.0002666

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006935, upper bound: 0.0007173
time: 1.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006924, upper bound: 0.0007169
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0038732, -0.0003915, -0.0038980, -0.0006038, -0.0025925, 0.0028539
1: -0.0040307, -0.0030490, -0.0040376, -0.0031089, -0.0007309, 0.0008046
2: 0.0088209, 0.0160635, 0.0087693, 0.0156219, -0.0053929, 0.0059367
3: 0.0027946, 0.0037530, 0.0027878, 0.0036946, -0.0007137, 0.0007856
4: -0.0059130, -0.0005003, -0.0055830, -0.0004617, -0.0044367, 0.0040303
5: 0.9938633, 0.9953673, 0.9939551, 0.9953780, -0.0012326, 0.0011198
6: 0.0023135, 0.0036785, 0.0023967, 0.0036882, -0.0011189, 0.0010164
7: -0.0147479, -0.0096540, -0.0144373, -0.0096177, -0.0041754, 0.0037930
8: -0.0016792, 0.0022855, -0.0017074, 0.0020437, -0.0029521, 0.0032497
9: -0.0042069, -0.0038649, -0.0041861, -0.0038624, -0.0002804, 0.0002547

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006961, upper bound: 0.0007169
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007169
time: 2.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038732, -0.0003915, -0.0038938, -0.0003890, -0.0027569, 0.0028278
1: -0.0040307, -0.0030490, -0.0040365, -0.0030483, -0.0007773, 0.0007973
2: 0.0088209, 0.0160635, 0.0087779, 0.0160687, -0.0057349, 0.0058823
3: 0.0027946, 0.0037530, 0.0027889, 0.0037537, -0.0007589, 0.0007784
4: -0.0059130, -0.0005003, -0.0059169, -0.0004682, -0.0043961, 0.0042859
5: 0.9938633, 0.9953673, 0.9938624, 0.9953762, -0.0012214, 0.0011908
6: 0.0023135, 0.0036785, 0.0023125, 0.0036866, -0.0011086, 0.0010808
7: -0.0147479, -0.0096540, -0.0147516, -0.0096237, -0.0041372, 0.0040335
8: -0.0016792, 0.0022855, -0.0017027, 0.0022883, -0.0031393, 0.0032200
9: -0.0042069, -0.0038649, -0.0042072, -0.0038628, -0.0002778, 0.0002708

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006961, upper bound: 0.0007169
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007173
time: 2.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0037573, -0.0001683, -0.0038428, -0.0006319, -0.0025748, 0.0029993
1: -0.0039980, -0.0029861, -0.0040221, -0.0031168, -0.0007259, 0.0008456
2: 0.0090618, 0.0165277, 0.0088841, 0.0155633, -0.0053560, 0.0062391
3: 0.0028265, 0.0038145, 0.0028030, 0.0036869, -0.0007088, 0.0008256
4: -0.0062599, -0.0006804, -0.0055392, -0.0005476, -0.0046627, 0.0040028
5: 0.9937670, 0.9953172, 0.9939673, 0.9953541, -0.0012954, 0.0011121
6: 0.0022260, 0.0036331, 0.0024078, 0.0036666, -0.0011759, 0.0010094
7: -0.0150744, -0.0098235, -0.0143961, -0.0096985, -0.0043881, 0.0037671
8: -0.0015472, 0.0025396, -0.0016445, 0.0020117, -0.0029319, 0.0034153
9: -0.0042288, -0.0038762, -0.0041833, -0.0038679, -0.0002947, 0.0002530

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006915, upper bound: 0.0007212
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006915, upper bound: 0.0007205
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0037573, -0.0001683, -0.0038431, -0.0004145, -0.0026197, 0.0029683
1: -0.0039980, -0.0029861, -0.0040222, -0.0030555, -0.0007386, 0.0008369
2: 0.0090618, 0.0165277, 0.0088834, 0.0160156, -0.0054494, 0.0061746
3: 0.0028265, 0.0038145, 0.0028029, 0.0037467, -0.0007211, 0.0008171
4: -0.0062599, -0.0006804, -0.0058772, -0.0005470, -0.0046145, 0.0040726
5: 0.9937670, 0.9953172, 0.9938735, 0.9953542, -0.0012821, 0.0011315
6: 0.0022260, 0.0036331, 0.0023225, 0.0036667, -0.0011637, 0.0010270
7: -0.0150744, -0.0098235, -0.0147142, -0.0096980, -0.0043428, 0.0038327
8: -0.0015472, 0.0025396, -0.0016449, 0.0022592, -0.0029830, 0.0033800
9: -0.0042288, -0.0038762, -0.0042047, -0.0038678, -0.0002916, 0.0002574

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006915, upper bound: 0.0007212
time: 2.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006915, upper bound: 0.0007204
time: 2.02 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037884, -0.0001520, -0.0038428, -0.0006335, -0.0026237, 0.0030132
1: -0.0040068, -0.0029815, -0.0040221, -0.0031173, -0.0007397, 0.0008495
2: 0.0089972, 0.0165617, 0.0088840, 0.0155601, -0.0054578, 0.0062682
3: 0.0028179, 0.0038190, 0.0028030, 0.0036864, -0.0007223, 0.0008295
4: -0.0062853, -0.0006321, -0.0055368, -0.0005475, -0.0046844, 0.0040789
5: 0.9937599, 0.9953306, 0.9939680, 0.9953541, -0.0013015, 0.0011332
6: 0.0022196, 0.0036453, 0.0024084, 0.0036666, -0.0011813, 0.0010286
7: -0.0150983, -0.0097780, -0.0143939, -0.0096984, -0.0044086, 0.0038387
8: -0.0015827, 0.0025582, -0.0016446, 0.0020099, -0.0029876, 0.0034312
9: -0.0042304, -0.0038732, -0.0041831, -0.0038678, -0.0002960, 0.0002578

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007209
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007212
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037884, -0.0001520, -0.0038431, -0.0004156, -0.0026585, 0.0029826
1: -0.0040068, -0.0029815, -0.0040222, -0.0030558, -0.0007495, 0.0008409
2: 0.0089972, 0.0165617, 0.0088834, 0.0160133, -0.0055302, 0.0062044
3: 0.0028179, 0.0038190, 0.0028029, 0.0037464, -0.0007318, 0.0008210
4: -0.0062853, -0.0006321, -0.0058755, -0.0005470, -0.0046367, 0.0041329
5: 0.9937599, 0.9953306, 0.9938738, 0.9953542, -0.0012882, 0.0011482
6: 0.0022196, 0.0036453, 0.0023230, 0.0036667, -0.0011693, 0.0010423
7: -0.0150983, -0.0097780, -0.0147126, -0.0096980, -0.0043637, 0.0038895
8: -0.0015827, 0.0025582, -0.0016449, 0.0022580, -0.0030272, 0.0033963
9: -0.0042304, -0.0038732, -0.0042045, -0.0038678, -0.0002930, 0.0002612

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007205
time: 2.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006940, upper bound: 0.0007205
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0037576, -0.0001652, -0.0038980, -0.0006003, -0.0026021, 0.0031041
1: -0.0039981, -0.0029852, -0.0040377, -0.0031079, -0.0007336, 0.0008752
2: 0.0090612, 0.0165341, 0.0087692, 0.0156290, -0.0054129, 0.0064572
3: 0.0028264, 0.0038153, 0.0027878, 0.0036955, -0.0007163, 0.0008545
4: -0.0062647, -0.0006799, -0.0055883, -0.0004617, -0.0048257, 0.0040452
5: 0.9937658, 0.9953174, 0.9939536, 0.9953780, -0.0013407, 0.0011239
6: 0.0022248, 0.0036332, 0.0023954, 0.0036882, -0.0012170, 0.0010202
7: -0.0150789, -0.0098230, -0.0144423, -0.0096177, -0.0045416, 0.0038070
8: -0.0015476, 0.0025431, -0.0017074, 0.0020476, -0.0029630, 0.0035347
9: -0.0042291, -0.0038762, -0.0041864, -0.0038624, -0.0003050, 0.0002556

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006724, upper bound: 0.0006978
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006696, upper bound: 0.0006978
time: 1.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0037576, -0.0001652, -0.0038939, -0.0003862, -0.0026499, 0.0030754
1: -0.0039981, -0.0029852, -0.0040365, -0.0030476, -0.0007471, 0.0008671
2: 0.0090612, 0.0165341, 0.0087777, 0.0160744, -0.0055123, 0.0063975
3: 0.0028264, 0.0038153, 0.0027889, 0.0037545, -0.0007295, 0.0008466
4: -0.0062647, -0.0006799, -0.0059211, -0.0004680, -0.0047811, 0.0041196
5: 0.9937658, 0.9953174, 0.9938611, 0.9953762, -0.0013283, 0.0011445
6: 0.0022248, 0.0036332, 0.0023114, 0.0036866, -0.0012057, 0.0010389
7: -0.0150789, -0.0098230, -0.0147556, -0.0096236, -0.0044995, 0.0038770
8: -0.0015476, 0.0025431, -0.0017028, 0.0022914, -0.0030175, 0.0035020
9: -0.0042291, -0.0038762, -0.0042074, -0.0038628, -0.0003021, 0.0002603

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006724, upper bound: 0.0006978
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006696, upper bound: 0.0006983
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037887, -0.0001494, -0.0038980, -0.0006038, -0.0025218, 0.0031179
1: -0.0040068, -0.0029808, -0.0040376, -0.0031089, -0.0007110, 0.0008791
2: 0.0089967, 0.0165671, 0.0087693, 0.0156219, -0.0052458, 0.0064859
3: 0.0028179, 0.0038197, 0.0027878, 0.0036946, -0.0006942, 0.0008583
4: -0.0062893, -0.0006317, -0.0055830, -0.0004617, -0.0048471, 0.0039204
5: 0.9937589, 0.9953306, 0.9939551, 0.9953780, -0.0013467, 0.0010892
6: 0.0022186, 0.0036454, 0.0023967, 0.0036882, -0.0012224, 0.0009887
7: -0.0151021, -0.0097776, -0.0144373, -0.0096177, -0.0045617, 0.0036895
8: -0.0015829, 0.0025611, -0.0017074, 0.0020437, -0.0028716, 0.0035504
9: -0.0042307, -0.0038732, -0.0041861, -0.0038624, -0.0003063, 0.0002477

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006743, upper bound: 0.0006978
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006986
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037887, -0.0001494, -0.0038938, -0.0003890, -0.0026872, 0.0030893
1: -0.0040068, -0.0029808, -0.0040365, -0.0030483, -0.0007576, 0.0008710
2: 0.0089967, 0.0165671, 0.0087779, 0.0160687, -0.0055898, 0.0064265
3: 0.0028179, 0.0038197, 0.0027889, 0.0037537, -0.0007397, 0.0008504
4: -0.0062893, -0.0006317, -0.0059169, -0.0004682, -0.0048027, 0.0041775
5: 0.9937589, 0.9953306, 0.9938624, 0.9953762, -0.0013343, 0.0011606
6: 0.0022186, 0.0036454, 0.0023125, 0.0036866, -0.0012112, 0.0010535
7: -0.0151021, -0.0097776, -0.0147516, -0.0096237, -0.0045199, 0.0039315
8: -0.0015829, 0.0025611, -0.0017027, 0.0022883, -0.0030599, 0.0035178
9: -0.0042307, -0.0038732, -0.0042072, -0.0038628, -0.0003035, 0.0002640

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006739, upper bound: 0.0006978
time: 2.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006978
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038632, -0.0004944, -0.0037597, -0.0003904, -0.0028053, 0.0026352
1: -0.0040278, -0.0030780, -0.0039987, -0.0030487, -0.0007909, 0.0007430
2: 0.0088417, 0.0158494, 0.0090569, 0.0160657, -0.0058357, 0.0054817
3: 0.0027974, 0.0037247, 0.0028258, 0.0037533, -0.0007723, 0.0007254
4: -0.0057530, -0.0005159, -0.0059146, -0.0006767, -0.0040967, 0.0043612
5: 0.9939078, 0.9953629, 0.9938630, 0.9953182, -0.0011382, 0.0012117
6: 0.0023538, 0.0036746, 0.0023131, 0.0036340, -0.0010331, 0.0010998
7: -0.0145973, -0.0096686, -0.0147495, -0.0098200, -0.0038554, 0.0041044
8: -0.0016678, 0.0021683, -0.0015500, 0.0022867, -0.0031944, 0.0030007
9: -0.0041968, -0.0038658, -0.0042070, -0.0038760, -0.0002589, 0.0002756

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006884, upper bound: 0.0007134
time: 2.15 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006884, upper bound: 0.0007209
time: 2.12 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038632, -0.0004944, -0.0038102, -0.0003593, -0.0028693, 0.0027228
1: -0.0040278, -0.0030780, -0.0040129, -0.0030400, -0.0008089, 0.0007677
2: 0.0088417, 0.0158494, 0.0089518, 0.0161304, -0.0059686, 0.0056640
3: 0.0027974, 0.0037247, 0.0028119, 0.0037619, -0.0007899, 0.0007495
4: -0.0057530, -0.0005159, -0.0059630, -0.0005981, -0.0042329, 0.0044606
5: 0.9939078, 0.9953629, 0.9938495, 0.9953401, -0.0011760, 0.0012393
6: 0.0023538, 0.0036746, 0.0023009, 0.0036538, -0.0010675, 0.0011249
7: -0.0145973, -0.0096686, -0.0147950, -0.0097461, -0.0039836, 0.0041979
8: -0.0016678, 0.0021683, -0.0016075, 0.0023221, -0.0032672, 0.0031005
9: -0.0041968, -0.0038658, -0.0042101, -0.0038710, -0.0002675, 0.0002819

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.31 + 595.76 = 600.07 seconds
