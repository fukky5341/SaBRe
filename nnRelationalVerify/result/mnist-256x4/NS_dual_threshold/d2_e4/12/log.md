## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 16.772132178899998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205)
1: (-11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597)
2: (-20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066)
3: (-17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639)
4: (-17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085)
5: (-13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499)
6: (-14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247)
7: (-15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786)
8: (-18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871)
9: (-12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.14 + 6.13 = 8.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -16.7889211, upper bound: 16.7889210

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866950, upper bound: 16.7857822
time: 3.97 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422
time: 5.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.35 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 2, lower bound: -16.7866950, upper bound: 16.7857822
NS_B2, status: Status.UNKNOWN, split count: 1, time: 9.35
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -14.0804739, 10.5467358, -13.1553812, 9.8612204, -23.9416885, 23.7021160
1: -11.1800871, 9.2234268, -10.4290733, 8.6319809, -19.8120670, 19.6525002
2: -19.0600471, 5.7586088, -17.8343563, 5.3188634, -24.3789101, 23.5929642
3: -16.4190426, 7.5271597, -15.3230782, 7.0573559, -23.4763947, 22.8502369
4: -16.3604507, 10.2050457, -15.2709064, 9.5472775, -25.9077282, 25.4759502
5: -12.5420485, 10.3862896, -11.7182293, 9.7029676, -22.2450142, 22.1045151
6: -13.4173231, 11.3542366, -12.5382681, 10.5985508, -24.0158730, 23.8925056
7: -14.6259489, 10.9898577, -13.6114149, 10.2653217, -24.8912678, 24.6012726
8: -16.8323956, 10.1548290, -15.7228317, 9.5185871, -26.3509827, 25.8776569
9: -11.9932089, 13.8124647, -11.1921597, 12.9067459, -24.8999557, 25.0046196

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7847133, upper bound: 16.7834235
time: 7.80 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834068
time: 9.24 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -15.0813789, 11.2827454, -14.8880434, 11.1396084, -26.2209873, 26.1707878
1: -11.9968710, 9.8736877, -11.8387756, 9.7469139, -21.7437859, 21.7124634
2: -20.3788204, 6.2278857, -20.1285057, 6.1339169, -26.5127335, 26.3563919
3: -17.6044922, 8.0591736, -17.3768883, 7.9534426, -25.5579338, 25.4360619
4: -17.5211048, 10.9149046, -17.2994347, 10.7768316, -28.2979355, 28.2143402
5: -13.4232674, 11.1254854, -13.2537823, 10.9827633, -24.4060307, 24.3792686
6: -14.3546829, 12.1642437, -14.1758566, 12.0075998, -26.3622818, 26.3400993
7: -15.6885843, 11.7738943, -15.4852552, 11.6217508, -27.3103352, 27.2591476
8: -18.0344696, 10.8528175, -17.8031006, 10.7171688, -28.7516365, 28.6559181
9: -12.8600473, 14.7708082, -12.6916313, 14.5878029, -27.4478493, 27.4624405

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857823, upper bound: 16.7866950
time: 6.93 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857823, upper bound: 16.7888424
time: 4.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.92 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -16.7847133, upper bound: 16.7834235
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834068
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -16.7857823, upper bound: 16.7866950
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 2, lower bound: -16.7857823, upper bound: 16.7888424

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -12.2578859, 9.2050505, -12.6249704, 9.4701347, -21.7280197, 21.8300209
1: -9.7015657, 8.0464706, -10.0005665, 8.2907162, -17.9922829, 18.0470371
2: -16.6006622, 5.0137978, -17.1178532, 5.1050305, -21.7056904, 22.1316509
3: -14.2376699, 6.5809684, -14.6887798, 6.7827706, -21.0204391, 21.2697487
4: -14.2258568, 8.9182339, -14.6500359, 9.1726456, -23.3985023, 23.5682678
5: -10.9308529, 9.0428495, -11.2485857, 9.3123894, -20.2432423, 20.2914352
6: -11.6919565, 9.8947926, -12.0356493, 10.1745949, -21.8665485, 21.9304390
7: -12.7063494, 9.5942497, -13.0528154, 9.8597803, -22.5661278, 22.6470642
8: -14.6469030, 8.8908749, -15.0878754, 9.1507521, -23.7976532, 23.9787502
9: -10.4393396, 12.0285339, -10.7408810, 12.3870487, -22.8263874, 22.7694149

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834066
time: 3.76 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834066
time: 7.36 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -12.6095963, 9.4709768, -12.2588453, 9.1980801, -21.8076763, 21.7298203
1: -9.9865379, 8.2761841, -9.7035675, 8.0546589, -18.0411949, 17.9797497
2: -17.0905762, 5.1727180, -16.6215458, 4.9504685, -22.0410442, 21.7942638
3: -14.6538153, 6.7659736, -14.2512999, 6.5903635, -21.2441788, 21.0172729
4: -14.6364813, 9.1715488, -14.2223082, 8.9133282, -23.5498085, 23.3938560
5: -11.2431507, 9.3053932, -10.9230938, 9.0421600, -20.2853088, 20.2284851
6: -12.0364037, 10.1768351, -11.6859531, 9.8819361, -21.9183369, 21.8627892
7: -13.0725689, 9.8703709, -12.6659374, 9.5774994, -22.6500664, 22.5363083
8: -15.0800247, 9.1537361, -14.6476374, 8.8933020, -23.9733238, 23.8013725
9: -10.7436628, 12.3754072, -10.4285383, 12.0267935, -22.7704563, 22.8039455

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834066
time: 3.03 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834067
time: 3.56 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -13.1553812, 9.8612204, -14.8880434, 11.1396084, -24.2949886, 24.7492638
1: -10.4290733, 8.6319809, -11.8387756, 9.7469139, -20.1759872, 20.4707565
2: -17.8343563, 5.3188634, -20.1285057, 6.1339169, -23.9682732, 25.4473686
3: -15.3230782, 7.0573559, -17.3768883, 7.9534426, -23.2765198, 24.4342442
4: -15.2709064, 9.5472775, -17.2994347, 10.7768316, -26.0477371, 26.8467121
5: -11.7182293, 9.7029676, -13.2537823, 10.9827633, -22.7009869, 22.9567490
6: -12.5382681, 10.5985508, -14.1758566, 12.0075998, -24.5458679, 24.7744064
7: -13.6114149, 10.2653217, -15.4852552, 11.6217508, -25.2331657, 25.7505760
8: -15.7228317, 9.5185871, -17.8031006, 10.7171688, -26.4399986, 27.3216877
9: -11.1921597, 12.9067459, -12.6916313, 14.5878029, -25.7799606, 25.5983753

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856505, upper bound: 16.7866108
time: 3.18 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856519, upper bound: 16.7866950
time: 3.16 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -14.8880434, 11.1396084, -14.8880434, 11.1396084, -26.0276527, 26.0276527
1: -11.8387756, 9.7469139, -11.8387756, 9.7469139, -21.5856895, 21.5856895
2: -20.1285057, 6.1339169, -20.1285057, 6.1339169, -26.2624207, 26.2624226
3: -17.3768883, 7.9534426, -17.3768883, 7.9534426, -25.3303299, 25.3303299
4: -17.2994347, 10.7768316, -17.2994347, 10.7768316, -28.0762672, 28.0762672
5: -13.2537823, 10.9827633, -13.2537823, 10.9827633, -24.2365437, 24.2365456
6: -14.1758566, 12.0075998, -14.1758566, 12.0075998, -26.1834564, 26.1834564
7: -15.4852552, 11.6217508, -15.4852552, 11.6217508, -27.1070061, 27.1070061
8: -17.8031006, 10.7171688, -17.8031006, 10.7171688, -28.5202694, 28.5202694
9: -12.6916313, 14.5878029, -12.6916313, 14.5878029, -27.2794304, 27.2794304

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7883528
time: 3.60 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831626
time: 5.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 11.15 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834066
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834066
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834066
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7846180, upper bound: 16.7834067
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7856505, upper bound: 16.7866108
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7856519, upper bound: 16.7866950
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7883528
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 11.15
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831626

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.2578859, 9.2050505, -11.3294468, 8.5095978, -20.7674828, 20.5344963
1: -9.7015657, 8.0464706, -8.9493618, 7.4590425, -17.1606083, 16.9958324
2: -16.6006622, 5.0137978, -15.3626728, 4.5824523, -21.1831150, 20.3764706
3: -14.2376699, 6.5809684, -13.1436024, 6.1100559, -20.3477249, 19.7245693
4: -14.2258568, 8.9182339, -13.1317701, 8.2561541, -22.4820099, 22.0500031
5: -10.9308529, 9.0428495, -10.0966663, 8.3574715, -19.2883244, 19.1395149
6: -11.6919565, 9.8947926, -10.8012285, 9.1402321, -20.8321877, 20.6960163
7: -12.7063494, 9.5942497, -11.6853247, 8.8681126, -21.5744629, 21.2795715
8: -14.6469030, 8.8908749, -13.5351000, 8.2485962, -22.8954964, 22.4259739
9: -10.4393396, 12.0285339, -9.6386185, 11.1134672, -21.5528069, 21.6671524

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7832489
time: 3.01 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7834236
time: 6.09 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.2578859, 9.2050505, -11.7013988, 8.7869463, -21.0448322, 20.9064484
1: -9.7015657, 8.0464706, -9.2480412, 7.7003651, -17.4019299, 17.2945118
2: -16.6006622, 5.0137978, -15.8788128, 4.7413774, -21.3420391, 20.8926105
3: -14.2376699, 6.5809684, -13.5779161, 6.3017569, -20.5394249, 20.1588840
4: -14.2258568, 8.9182339, -13.5624933, 8.5213795, -22.7472363, 22.4807281
5: -10.9308529, 9.0428495, -10.4246883, 8.6319876, -19.5628395, 19.4675369
6: -11.6919565, 9.8947926, -11.1627178, 9.4358931, -21.1278496, 21.0575104
7: -12.7063494, 9.5942497, -12.0726881, 9.1550150, -21.8613644, 21.6669369
8: -14.6469030, 8.8908749, -13.9915066, 8.5206900, -23.1675892, 22.8823814
9: -10.4393396, 12.0285339, -9.9539185, 11.4785490, -21.9178886, 21.9824524

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7832488
time: 2.96 seconds

## Relational analysis of NS_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7834236
time: 4.37 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.6095963, 9.4709768, -11.3294468, 8.5095978, -21.1191940, 20.8004227
1: -9.9865379, 8.2761841, -8.9493618, 7.4590425, -17.4455795, 17.2255421
2: -17.0905762, 5.1727180, -15.3626728, 4.5824523, -21.6730289, 20.5353889
3: -14.6538153, 6.7659736, -13.1436024, 6.1100559, -20.7638702, 19.9095745
4: -14.6364813, 9.1715488, -13.1317701, 8.2561541, -22.8926353, 22.3033180
5: -11.2431507, 9.3053932, -10.0966663, 8.3574715, -19.6006203, 19.4020576
6: -12.0364037, 10.1768351, -10.8012285, 9.1402321, -21.1766357, 20.9780598
7: -13.0725689, 9.8703709, -11.6853247, 8.8681126, -21.9406796, 21.5556946
8: -15.0800247, 9.1537361, -13.5351000, 8.2485962, -23.3286209, 22.6888332
9: -10.7436628, 12.3754072, -9.6386185, 11.1134672, -21.8571301, 22.0140228

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831625
time: 3.03 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831625
time: 3.61 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.6095963, 9.4709768, -11.7013988, 8.7869463, -21.3965397, 21.1723747
1: -9.9865379, 8.2761841, -9.2480412, 7.7003651, -17.6869011, 17.5242214
2: -17.0905762, 5.1727180, -15.8788128, 4.7413774, -21.8319530, 21.0515308
3: -14.6538153, 6.7659736, -13.5779161, 6.3017569, -20.9555721, 20.3438892
4: -14.6364813, 9.1715488, -13.5624933, 8.5213795, -23.1578598, 22.7340431
5: -11.2431507, 9.3053932, -10.4246883, 8.6319876, -19.8751354, 19.7300797
6: -12.0364037, 10.1768351, -11.1627178, 9.4358931, -21.4722977, 21.3395500
7: -13.0725689, 9.8703709, -12.0726881, 9.1550150, -22.2275810, 21.9430580
8: -15.0800247, 9.1537361, -13.9915066, 8.5206900, -23.6007099, 23.1452408
9: -10.7436628, 12.3754072, -9.9539185, 11.4785490, -22.2222118, 22.3293228

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831626
time: 3.00 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7834066
time: 2.72 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -11.9465618, 8.9598589, -14.5912085, 10.9191694, -22.8657303, 23.5510674
1: -9.4561272, 7.8464103, -11.5983343, 9.5532713, -19.0093975, 19.4447441
2: -16.1669731, 4.7565565, -19.7291241, 5.9998860, -22.1668587, 24.4856796
3: -13.8956308, 6.4204049, -17.0260029, 7.7961001, -21.6917267, 23.4464054
4: -13.8736906, 8.6901970, -16.9568596, 10.5660925, -24.4397831, 25.6470566
5: -10.6566067, 8.8103752, -12.9929571, 10.7634306, -21.4200363, 21.8033333
6: -11.3812904, 9.6378384, -13.8958998, 11.7695112, -23.1508026, 23.5337372
7: -12.3452015, 9.3238211, -15.1746120, 11.3914728, -23.7366753, 24.4984322
8: -14.2464867, 8.6353340, -17.4448872, 10.5042591, -24.7507458, 26.0802212
9: -10.1612339, 11.7199535, -12.4372034, 14.3006649, -24.4618988, 24.1571579

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857685, upper bound: 16.7865938
time: 3.53 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857575, upper bound: 16.7865936
time: 4.43 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -12.9603901, 9.7170582, -14.8250895, 11.0926628, -24.0530529, 24.5421448
1: -10.2720270, 8.5059757, -11.7878675, 9.7057619, -19.9777889, 20.2938423
2: -17.5687256, 5.2376623, -20.0443592, 6.1054668, -23.6741886, 25.2820187
3: -15.0909796, 6.9561176, -17.3023720, 7.9198275, -23.0108070, 24.2584896
4: -15.0437145, 9.4091997, -17.2269402, 10.7320652, -25.7757797, 26.6361389
5: -11.5461197, 9.5592213, -13.1984186, 10.9362631, -22.4823837, 22.7576408
6: -12.3525496, 10.4429226, -14.1166897, 11.9570312, -24.3095779, 24.5596085
7: -13.4068880, 10.1157665, -15.4194756, 11.5728636, -24.9797516, 25.5352421
8: -15.4874554, 9.3806763, -17.7275391, 10.6721306, -26.1595860, 27.1082153
9: -11.0263796, 12.7155571, -12.6375952, 14.5269613, -25.5533352, 25.3531532

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7834238, upper bound: 16.7847130
time: 23.62 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846180
time: 4.12 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -13.0021305, 9.7564154, -14.3336344, 10.7326059, -23.7347374, 24.0900497
1: -10.3025303, 8.5233068, -11.3872061, 9.3862820, -19.6888123, 19.9105129
2: -17.6045723, 5.3444290, -19.3887844, 5.9006977, -23.5052700, 24.7332134
3: -15.1256447, 6.9662085, -16.7162228, 7.6629372, -22.7885818, 23.6824303
4: -15.0982275, 9.4448204, -16.6536808, 10.3837538, -25.4819813, 26.0984993
5: -11.5900106, 9.5939112, -12.7647734, 10.5748234, -22.1648331, 22.3586845
6: -12.3999119, 10.4958715, -13.6546822, 11.5628996, -23.9628105, 24.1505547
7: -13.4994879, 10.1725845, -14.9029150, 11.1950073, -24.6944942, 25.0755005
8: -15.5442801, 9.4097328, -17.1396790, 10.3320303, -25.8763103, 26.5494118
9: -11.0783777, 12.7566023, -12.2171202, 14.0500603, -25.1284370, 24.9737206

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883420
time: 3.29 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883417
time: 3.08 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -13.3641338, 10.0305777, -13.9555798, 10.4564524, -23.8205872, 23.9861565
1: -10.5947323, 8.7590675, -11.0767250, 9.1383848, -19.7331142, 19.8357887
2: -18.1034241, 5.5068827, -18.8872776, 5.7291307, -23.8325539, 24.3941555
3: -15.5508480, 7.1607552, -16.2632637, 7.4601860, -23.0110340, 23.4240189
4: -15.5178633, 9.7034388, -16.2144108, 10.1166077, -25.6344700, 25.9178505
5: -11.9108944, 9.8634520, -12.4323692, 10.2957935, -22.2066879, 22.2958202
6: -12.7511435, 10.7846441, -13.3004646, 11.2586699, -24.0098114, 24.0851097
7: -13.8768320, 10.4565668, -14.5043736, 10.9006348, -24.7774639, 24.9609413
8: -15.9861116, 9.6781807, -16.6855202, 10.0685787, -26.0546913, 26.3637009
9: -11.3907738, 13.1116152, -11.8904343, 13.6866016, -25.0773735, 25.0020485

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 97

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883418
time: 4.56 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883419
time: 2.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 9.25 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7832489
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7834236
NS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7832488
NS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7838436, upper bound: 16.7834236
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831625
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831625
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7831626
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7831626, upper bound: 16.7834066
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7857685, upper bound: 16.7865938
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7857575, upper bound: 16.7865936
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7834238, upper bound: 16.7847130
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846180
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883420
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883417
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883418
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.25
Output dim: 2, lower bound: -16.7883422, upper bound: 16.7883419

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.3069611, 8.4929981, -11.3294468, 8.5095978, -19.8165569, 19.8224430
1: -8.9311705, 7.4445481, -8.9493618, 7.4590425, -16.3902130, 16.3939056
2: -15.3321781, 4.5738339, -15.3626728, 4.5824523, -19.9146309, 19.9365063
3: -13.1168795, 6.0984936, -13.1436024, 6.1100559, -19.2269325, 19.2420959
4: -13.1054134, 8.2402487, -13.1317701, 8.2561541, -21.3615685, 21.3720188
5: -10.0767269, 8.3409386, -10.0966663, 8.3574715, -18.4341965, 18.4376049
6: -10.7796898, 9.1223259, -10.8012285, 9.1402321, -19.9199219, 19.9235535
7: -11.6618357, 8.8510656, -11.6853247, 8.8681126, -20.5299473, 20.5363884
8: -13.5079260, 8.2329311, -13.5351000, 8.2485962, -21.7565231, 21.7680302
9: -9.6195812, 11.0913677, -9.6386185, 11.1134672, -20.7330475, 20.7299843

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B1_A1_A1

### Relational analysis result of NS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7844141, upper bound: 16.7847341
time: 2.27 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2

### Relational analysis result of NS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855355, upper bound: 16.7855354
time: 2.89 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.9916124, 9.7492447, -11.3294468, 8.5095978, -21.5012093, 21.0786896
1: -10.2961349, 8.5171452, -8.9493618, 7.4590425, -17.7551746, 17.4665051
2: -17.5900211, 5.3309698, -15.3626728, 4.5824523, -22.1724739, 20.6936417
3: -15.1162367, 6.9617815, -13.1436024, 6.1100559, -21.2262917, 20.1053848
4: -15.0881214, 9.4391270, -13.1317701, 8.2561541, -23.3442764, 22.5708961
5: -11.5820236, 9.5861473, -10.0966663, 8.3574715, -19.9394951, 19.6828136
6: -12.3906593, 10.4864187, -10.8012285, 9.1402321, -21.5308914, 21.2876453
7: -13.4893522, 10.1611195, -11.6853247, 8.8681126, -22.3574638, 21.8464432
8: -15.5283356, 9.4040041, -13.5351000, 8.2485962, -23.7769279, 22.9391041
9: -11.0699492, 12.7496281, -9.6386185, 11.1134672, -22.1834145, 22.3882446

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7847341, upper bound: 16.7845596
time: 2.84 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855355, upper bound: 16.7855352
time: 2.58 seconds

## BFS NS instance: NS_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -11.3069611, 8.4929981, -11.7013988, 8.7869463, -20.0939045, 20.1943951
1: -8.9311705, 7.4445481, -9.2480412, 7.7003651, -16.6315327, 16.6925850
2: -15.3321781, 4.5738339, -15.8788128, 4.7413774, -20.0735550, 20.4526463
3: -13.1168795, 6.0984936, -13.5779161, 6.3017569, -19.4186363, 19.6764069
4: -13.1054134, 8.2402487, -13.5624933, 8.5213795, -21.6267929, 21.8027420
5: -10.0767269, 8.3409386, -10.4246883, 8.6319876, -18.7087135, 18.7656250
6: -10.7796898, 9.1223259, -11.1627178, 9.4358931, -20.2155800, 20.2850437
7: -11.6618357, 8.8510656, -12.0726881, 9.1550150, -20.8168507, 20.9237518
8: -13.5079260, 8.2329311, -13.9915066, 8.5206900, -22.0286160, 22.2244377
9: -9.6195812, 11.0913677, -9.9539185, 11.4785490, -21.0981255, 21.0452843

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7831358, upper bound: 16.7827832
time: 3.24 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
time: 3.20 seconds

## BFS NS instance: NS_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.9916124, 9.7492447, -11.7013988, 8.7869463, -21.7785587, 21.4506416
1: -10.2961349, 8.5171452, -9.2480412, 7.7003651, -17.9964962, 17.7651844
2: -17.5900211, 5.3309698, -15.8788128, 4.7413774, -22.3313980, 21.2097816
3: -15.1162367, 6.9617815, -13.5779161, 6.3017569, -21.4179935, 20.5396976
4: -15.0881214, 9.4391270, -13.5624933, 8.5213795, -23.6094971, 23.0016212
5: -11.5820236, 9.5861473, -10.4246883, 8.6319876, -20.2140102, 20.0108337
6: -12.3906593, 10.4864187, -11.1627178, 9.4358931, -21.8265533, 21.6491356
7: -13.4893522, 10.1611195, -12.0726881, 9.1550150, -22.6443672, 22.2338066
8: -15.5283356, 9.4040041, -13.9915066, 8.5206900, -24.0490208, 23.3955116
9: -11.0699492, 12.7496281, -9.9539185, 11.4785490, -22.5484962, 22.7035446

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7831358, upper bound: 16.7827833
time: 4.28 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
time: 3.29 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.7024851, 8.7881804, -11.3294468, 8.5095978, -20.2120819, 20.1176262
1: -9.2491646, 7.7013502, -8.9493618, 7.4590425, -16.7082062, 16.6507092
2: -15.8805971, 4.7439752, -15.3626728, 4.5824523, -20.4630470, 20.1066475
3: -13.5801210, 6.3034339, -13.1436024, 6.1100559, -19.6901741, 19.4470310
4: -13.5635386, 8.5223713, -13.1317701, 8.2561541, -21.8196926, 21.6541405
5: -10.4259796, 8.6330090, -10.0966663, 8.3574715, -18.7834492, 18.7296734
6: -11.1640100, 9.4368343, -10.8012285, 9.1402321, -20.3042412, 20.2380619
7: -12.0741358, 9.1566820, -11.6853247, 8.8681126, -20.9422455, 20.8420048
8: -13.9927425, 8.5221806, -13.5351000, 8.2485962, -22.2413368, 22.0572777
9: -9.9551935, 11.4798756, -9.6386185, 11.1134672, -21.0686607, 21.1184921

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7827833, upper bound: 16.7831358
time: 3.20 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822746, upper bound: 16.7829011
time: 3.13 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13.3445158, 10.0156116, -11.3294468, 8.5095978, -21.8541088, 21.3450584
1: -10.5815210, 8.7482738, -8.9493618, 7.4590425, -18.0405636, 17.6976337
2: -18.0815983, 5.4891710, -15.3626728, 4.5824523, -22.6640511, 20.8518429
3: -15.5335932, 7.1467004, -13.1436024, 6.1100559, -21.6436501, 20.2903004
4: -15.4993877, 9.6934795, -13.1317701, 8.2561541, -23.7555389, 22.8252487
5: -11.8953190, 9.8489895, -10.0966663, 8.3574715, -20.2527885, 19.9456558
6: -12.7351007, 10.7695675, -10.8012285, 9.1402321, -21.8753319, 21.5707951
7: -13.8565035, 10.4395599, -11.6853247, 8.8681126, -22.7246151, 22.1248817
8: -15.9612513, 9.6667576, -13.5351000, 8.2485962, -24.2098465, 23.2018566
9: -11.3752518, 13.0967770, -9.6386185, 11.1134672, -22.4887161, 22.7353935

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 74

## Relational analysis of NS_B1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7808025, upper bound: 16.7818917
time: 7.34 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7807555, upper bound: 16.7832986
time: 4.55 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.7024851, 8.7881804, -11.7013988, 8.7869463, -20.4894295, 20.4895763
1: -9.2491646, 7.7013502, -9.2480412, 7.7003651, -16.9495277, 16.9493885
2: -15.8805971, 4.7439752, -15.8788128, 4.7413774, -20.6219711, 20.6227875
3: -13.5801210, 6.3034339, -13.5779161, 6.3017569, -19.8818760, 19.8813477
4: -13.5635386, 8.5223713, -13.5624933, 8.5213795, -22.0849152, 22.0848637
5: -10.4259796, 8.6330090, -10.4246883, 8.6319876, -19.0579643, 19.0576935
6: -11.1640100, 9.4368343, -11.1627178, 9.4358931, -20.5998993, 20.5995483
7: -12.0741358, 9.1566820, -12.0726881, 9.1550150, -21.2291470, 21.2293682
8: -13.9927425, 8.5221806, -13.9915066, 8.5206900, -22.5134296, 22.5136833
9: -9.9551935, 11.4798756, -9.9539185, 11.4785490, -21.4337425, 21.4337921

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7825065, upper bound: 16.7822137
time: 2.57 seconds

## Relational analysis of NS_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7821715, upper bound: 16.7821712
time: 4.84 seconds

## BFS NS instance: NS_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.3445158, 10.0156116, -11.7013988, 8.7869463, -22.1314564, 21.7170105
1: -10.5815210, 8.7482738, -9.2480412, 7.7003651, -18.2818851, 17.9963131
2: -18.0815983, 5.4891710, -15.8788128, 4.7413774, -22.8229752, 21.3679848
3: -15.5335932, 7.1467004, -13.5779161, 6.3017569, -21.8353500, 20.7246151
4: -15.4993877, 9.6934795, -13.5624933, 8.5213795, -24.0207672, 23.2559719
5: -11.8953190, 9.8489895, -10.4246883, 8.6319876, -20.5273037, 20.2736759
6: -12.7351007, 10.7695675, -11.1627178, 9.4358931, -22.1709900, 21.9322853
7: -13.8565035, 10.4395599, -12.0726881, 9.1550150, -23.0115147, 22.5122452
8: -15.9612513, 9.6667576, -13.9915066, 8.5206900, -24.4819412, 23.6582642
9: -11.3752518, 13.0967770, -9.9539185, 11.4785490, -22.8537979, 23.0506935

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822138, upper bound: 16.7825065
time: 4.42 seconds

## Relational analysis of NS_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7821715, upper bound: 16.7821712
time: 2.78 seconds

## BFS NS instance: NS_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -11.4679775, 8.6040430, -14.3760567, 10.7626171, -22.2305946, 22.9800949
1: -9.0689049, 7.5381589, -11.4240770, 9.4141006, -18.4830055, 18.9622345
2: -15.5088615, 4.5574846, -19.4412518, 5.9040842, -21.4129448, 23.9987373
3: -13.3293390, 6.1722469, -16.7719307, 7.6818900, -21.0112286, 22.9441757
4: -13.3153906, 8.3512688, -16.7096024, 10.4144650, -23.7298546, 25.0608711
5: -10.2337875, 8.4570341, -12.8056087, 10.6043406, -20.8381271, 21.2626381
6: -10.9246407, 9.2572117, -13.6958799, 11.5981026, -22.5227432, 22.9530869
7: -11.8427610, 8.9553356, -14.9513435, 11.2250147, -23.0677757, 23.9066792
8: -13.6687775, 8.2947283, -17.1880531, 10.3521547, -24.0209312, 25.4827805
9: -9.7544451, 11.2499247, -12.2530260, 14.0947046, -23.8491497, 23.5029507

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 131

### Candidate
type: B, layer: 1, pos: 76

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: A, layer: 1, pos: 74

## Relational analysis of NS_B2_A1_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7852522, upper bound: 16.7856644
time: 2.99 seconds

## Relational analysis of NS_B2_A1_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7840938, upper bound: 16.7850504
time: 13.26 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -12.3924809, 9.2894917, -14.3815813, 10.7671890, -23.1596699, 23.6710720
1: -9.8175240, 8.1317406, -11.4285898, 9.4179821, -19.2355061, 19.5603294
2: -16.7681255, 4.9358602, -19.4495430, 5.9076452, -22.6757698, 24.3854027
3: -14.4262447, 6.6501236, -16.7789993, 7.6851640, -22.1114082, 23.4291229
4: -14.3966751, 9.0047617, -16.7165089, 10.4188118, -24.8154869, 25.7212715
5: -11.0525455, 9.1386509, -12.8108883, 10.6084299, -21.6609764, 21.9495373
6: -11.8049173, 9.9939775, -13.7016573, 11.6027899, -23.4077072, 23.6956348
7: -12.8168468, 9.6637192, -14.9576378, 11.2299061, -24.0467529, 24.6213531
8: -14.7804880, 8.9431190, -17.1954460, 10.3568916, -25.1373749, 26.1385632
9: -10.5401716, 12.1578493, -12.2581835, 14.1003180, -24.6404858, 24.4160328

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 174

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 76

### Candidate
type: B, layer: 1, pos: 131

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 76

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857575, upper bound: 16.7865936
time: 3.72 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857575, upper bound: 16.7865936
time: 3.98 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -12.4319019, 9.3267841, -12.9473667, 9.7158384, -22.1477375, 22.2741489
1: -9.8444824, 8.1660347, -10.2583961, 8.4878578, -18.3323402, 18.4244308
2: -16.8540268, 5.0249853, -17.5299969, 5.3206930, -22.1747169, 22.5549812
3: -14.4596691, 6.6823463, -15.0600681, 6.9374933, -21.3971634, 21.7424145
4: -14.4250469, 9.0357618, -15.0342245, 9.4061031, -23.8311462, 24.0699844
5: -11.0776033, 9.1700535, -11.5416307, 9.5533943, -20.6309948, 20.7116814
6: -11.8509064, 10.0206547, -12.3477316, 10.4519434, -22.3028488, 22.3683815
7: -12.8501682, 9.7114210, -13.4416733, 10.1302614, -22.9804287, 23.1530952
8: -14.8546505, 9.0137768, -15.4780149, 9.3706923, -24.2253399, 24.4917908
9: -10.5767517, 12.1973839, -11.0315189, 12.7031441, -23.2798958, 23.2289028

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_B2_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846180
time: 12.31 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846177
time: 4.72 seconds

## BFS NS instance: NS_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -12.0653820, 9.0543127, -13.3063374, 9.9876699, -22.0530510, 22.3606472
1: -9.5470448, 7.9299154, -10.5483294, 8.7217770, -18.2688217, 18.4782429
2: -16.3571529, 4.8701138, -18.0253353, 5.4814811, -21.8386307, 22.8954468
3: -14.0217018, 6.4896564, -15.4821873, 7.1299663, -21.1516647, 21.9718437
4: -13.9966803, 8.7761688, -15.4507570, 9.6628628, -23.6595421, 24.2269249
5: -10.7516479, 8.8994226, -11.8599596, 9.8207359, -20.5723820, 20.7593803
6: -11.5006685, 9.7277412, -12.6963501, 10.7384214, -22.2390862, 22.4240913
7: -12.4627237, 9.4289865, -13.8159199, 10.4119177, -22.8746414, 23.2449036
8: -14.4138622, 8.7559452, -15.9162655, 9.6371851, -24.0510483, 24.6722088
9: -10.2640629, 11.8365765, -11.3414011, 13.0555067, -23.3195686, 23.1779766

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846180
time: 5.75 seconds

## Relational analysis of NS_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846177
time: 3.86 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -13.0021305, 9.7564154, -13.0021305, 9.7564154, -22.7585449, 22.7585449
1: -10.3025303, 8.5233068, -10.3025303, 8.5233068, -18.8258343, 18.8258362
2: -17.6045723, 5.3444290, -17.6045723, 5.3444290, -22.9490013, 22.9490013
3: -15.1256447, 6.9662085, -15.1256447, 6.9662085, -22.0918503, 22.0918522
4: -15.0982275, 9.4448204, -15.0982275, 9.4448204, -24.5430431, 24.5430450
5: -11.5900106, 9.5939112, -11.5900106, 9.5939112, -21.1839218, 21.1839218
6: -12.3999119, 10.4958715, -12.3999119, 10.4958715, -22.8957825, 22.8957825
7: -13.4994879, 10.1725845, -13.4994879, 10.1725845, -23.6720657, 23.6720695
8: -15.5442801, 9.4097328, -15.5442801, 9.4097328, -24.9540138, 24.9540138
9: -11.0783777, 12.7566023, -11.0783777, 12.7566023, -23.8349800, 23.8349800

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7860246, upper bound: 16.7861699
time: 4.70 seconds

## Relational analysis of NS_B2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855291, upper bound: 16.7850737
time: 3.09 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -13.0021305, 9.7564154, -13.3641338, 10.0305777, -23.0327072, 23.1205482
1: -10.3025303, 8.5233068, -10.5947323, 8.7590675, -19.0615921, 19.1180363
2: -17.6045723, 5.3444290, -18.1034241, 5.5068827, -23.1114521, 23.4478531
3: -15.1256447, 6.9662085, -15.5508480, 7.1607552, -22.2863998, 22.5170555
4: -15.0982275, 9.4448204, -15.5178633, 9.7034388, -24.8016624, 24.9626846
5: -11.5900106, 9.5939112, -11.9108944, 9.8634520, -21.4534607, 21.5048065
6: -12.3999119, 10.4958715, -12.7511435, 10.7846441, -23.1845551, 23.2470131
7: -13.4994879, 10.1725845, -13.8768320, 10.4565668, -23.9560547, 24.0494156
8: -15.5442801, 9.4097328, -15.9861116, 9.6781807, -25.2224617, 25.3958435
9: -11.0783777, 12.7566023, -11.3907738, 13.1116152, -24.1899910, 24.1473713

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 249

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7860246, upper bound: 16.7861699
time: 5.99 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855291, upper bound: 16.7850737
time: 3.23 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -13.3641338, 10.0305777, -13.0021305, 9.7564154, -23.1205482, 23.0327072
1: -10.5947323, 8.7590675, -10.3025303, 8.5233068, -19.1180363, 19.0615921
2: -18.1034241, 5.5068827, -17.6045723, 5.3444290, -23.4478531, 23.1114521
3: -15.5508480, 7.1607552, -15.1256447, 6.9662085, -22.5170555, 22.2863998
4: -15.5178633, 9.7034388, -15.0982275, 9.4448204, -24.9626827, 24.8016624
5: -11.9108944, 9.8634520, -11.5900106, 9.5939112, -21.5048065, 21.4534607
6: -12.7511435, 10.7846441, -12.3999119, 10.4958715, -23.2470112, 23.1845551
7: -13.8768320, 10.4565668, -13.4994879, 10.1725845, -24.0494156, 23.9560547
8: -15.9861116, 9.6781807, -15.5442801, 9.4097328, -25.3958435, 25.2224617
9: -11.3907738, 13.1116152, -11.0783777, 12.7566023, -24.1473732, 24.1899929

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 249

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of NS_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7859150, upper bound: 16.7851457
time: 15.81 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850282, upper bound: 16.7850281
time: 2.26 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -13.3641338, 10.0305777, -13.3641338, 10.0305777, -23.3947105, 23.3947105
1: -10.5947323, 8.7590675, -10.5947323, 8.7590675, -19.3537960, 19.3537960
2: -18.1034241, 5.5068827, -18.1034241, 5.5068827, -23.6103058, 23.6103058
3: -15.5508480, 7.1607552, -15.5508480, 7.1607552, -22.7116032, 22.7116032
4: -15.5178633, 9.7034388, -15.5178633, 9.7034388, -25.2213020, 25.2213020
5: -11.9108944, 9.8634520, -11.9108944, 9.8634520, -21.7743454, 21.7743454
6: -12.7511435, 10.7846441, -12.7511435, 10.7846441, -23.5357857, 23.5357876
7: -13.8768320, 10.4565668, -13.8768320, 10.4565668, -24.3333988, 24.3333988
8: -15.9861116, 9.6781807, -15.9861116, 9.6781807, -25.6642914, 25.6642914
9: -11.3907738, 13.1116152, -11.3907738, 13.1116152, -24.5023804, 24.5023823

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of NS_B2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7851456, upper bound: 16.7859134
time: 3.99 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850282, upper bound: 16.7850283
time: 2.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 8.63 seconds
NS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7844141, upper bound: 16.7847341
NS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7855355, upper bound: 16.7855354
NS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7847341, upper bound: 16.7845596
NS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7855355, upper bound: 16.7855352
NS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7831358, upper bound: 16.7827832
NS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
NS_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7831358, upper bound: 16.7827833
NS_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
NS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7827833, upper bound: 16.7831358
NS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7822746, upper bound: 16.7829011
NS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7808025, upper bound: 16.7818917
NS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7807555, upper bound: 16.7832986
NS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7825065, upper bound: 16.7822137
NS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7821715, upper bound: 16.7821712
NS_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7822138, upper bound: 16.7825065
NS_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7821715, upper bound: 16.7821712
NS_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7852522, upper bound: 16.7856644
NS_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7840938, upper bound: 16.7850504
NS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7857575, upper bound: 16.7865936
NS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7857575, upper bound: 16.7865936
NS_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846180
NS_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846177
NS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846180
NS_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7834068, upper bound: 16.7846177
NS_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7860246, upper bound: 16.7861699
NS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7855291, upper bound: 16.7850737
NS_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7860246, upper bound: 16.7861699
NS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7855291, upper bound: 16.7850737
NS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7859150, upper bound: 16.7851457
NS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7850282, upper bound: 16.7850281
NS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7851456, upper bound: 16.7859134
NS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 8.63
Output dim: 2, lower bound: -16.7850282, upper bound: 16.7850283

## BFS NS instance: NS_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -11.5220623, 8.6484022, -10.8562918, 8.1559172, -19.6779785, 19.5046940
1: -9.1078768, 7.5727797, -8.5663376, 7.1514301, -16.2593079, 16.1391182
2: -15.6070452, 4.6270962, -14.7126179, 4.3778033, -19.9848480, 19.3397141
3: -13.3698902, 6.1952949, -12.5794106, 5.8599272, -19.2298164, 18.7747059
4: -13.3685122, 8.3896112, -12.5806389, 7.9198966, -21.2884083, 20.9702492
5: -10.2711134, 8.4961853, -9.6768055, 8.0072775, -18.2783909, 18.1729908
6: -10.9738007, 9.2958288, -10.3444586, 8.7627745, -19.7365761, 19.6402874
7: -11.8982601, 9.0034580, -11.1883631, 8.5010872, -20.3993473, 20.1918221
8: -13.7501888, 8.3586779, -12.9589720, 7.9085994, -21.6587887, 21.3176498
9: -9.7999134, 11.3015299, -9.2353420, 10.6473713, -20.4472847, 20.5368729

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 158

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7843787, upper bound: 16.7843787
time: 2.97 seconds

## Relational analysis of NS_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7843787, upper bound: 16.7847341
time: 2.35 seconds

## BFS NS instance: NS_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -11.1483316, 8.3742599, -11.3155575, 8.4992189, -19.6475506, 19.6898155
1: -8.8029184, 7.3407569, -8.9381275, 7.4499722, -16.2528915, 16.2788849
2: -15.1146421, 4.5022497, -15.3436470, 4.5762339, -19.6908760, 19.8458977
3: -12.9271250, 6.0133977, -13.1270037, 6.1026273, -19.0297527, 19.1403999
4: -12.9214010, 8.1274786, -13.1156645, 8.2462597, -21.1676598, 21.2431431
5: -9.9360142, 8.2235165, -10.0843229, 8.3472061, -18.2832203, 18.3078384
6: -10.6258030, 8.9958496, -10.7877626, 9.1291685, -19.7549706, 19.7836075
7: -11.4957123, 8.7273102, -11.6707869, 8.8573036, -20.3530159, 20.3980980
8: -13.3138580, 8.1178837, -13.5181313, 8.2385378, -21.5523949, 21.6360149
9: -9.4840927, 10.9350643, -9.6267719, 11.0997667, -20.5838585, 20.5618362

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 97

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7847341, upper bound: 16.7844141
time: 3.28 seconds

## Relational analysis of NS_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7847341, upper bound: 16.7855355
time: 2.47 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.5732851, 9.4373341, -11.5220623, 8.6484022, -21.2216854, 20.9593964
1: -9.9577265, 8.2447834, -9.1078768, 7.5727797, -17.5305061, 17.3526611
2: -17.0159206, 5.1465645, -15.6070452, 4.6270962, -21.6430168, 20.7536087
3: -14.6152945, 6.7396741, -13.3698902, 6.1952949, -20.8105888, 20.1095638
4: -14.6004343, 9.1409626, -13.3685122, 8.3896112, -22.9900455, 22.5094757
5: -11.2118044, 9.2756329, -10.2711134, 8.4961853, -19.7079887, 19.5467453
6: -11.9881916, 10.1518850, -10.9738007, 9.2958288, -21.2840195, 21.1256866
7: -13.0509462, 9.8349838, -11.8982601, 9.0034580, -22.0544014, 21.7332439
8: -15.0201454, 9.1011152, -13.7501888, 8.3586779, -23.3788223, 22.8512993
9: -10.7107592, 12.3379412, -9.7999134, 11.3015299, -22.0122890, 22.1378517

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855133, upper bound: 16.7845421
time: 5.24 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855133, upper bound: 16.7845596
time: 4.58 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.9787979, 9.7396908, -11.1613512, 8.3838758, -21.3626747, 20.9010429
1: -10.2857819, 8.5087566, -8.8134527, 7.3491502, -17.6349316, 17.3222084
2: -17.5724831, 5.3251791, -15.1322966, 4.5072412, -22.0797234, 20.4574738
3: -15.1008577, 6.9549046, -12.9426031, 6.0200872, -21.1209431, 19.8975067
4: -15.0732155, 9.4299889, -12.9366665, 8.1366844, -23.2098999, 22.3666553
5: -11.5706854, 9.5766344, -9.9475565, 8.2330914, -19.8037758, 19.5241909
6: -12.3782797, 10.4761677, -10.6382694, 9.0062180, -21.3844986, 21.1144371
7: -13.4759617, 10.1510916, -11.5093145, 8.7371788, -22.2131405, 21.6604061
8: -15.5127115, 9.3946686, -13.3295889, 8.1269417, -23.6396523, 22.7242584
9: -11.0589323, 12.7370167, -9.4951143, 10.9478617, -22.0067940, 22.2321281

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of NS_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7859130, upper bound: 16.7851953
time: 4.89 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7859130, upper bound: 16.7851952
time: 4.81 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.3003654, 8.4880543, -11.4387197, 8.5899477, -19.8903122, 19.9267731
1: -8.9258385, 7.4402390, -9.0359592, 7.5286975, -16.4545364, 16.4761982
2: -15.3230743, 4.5707579, -15.5153294, 4.6186676, -19.9417419, 20.0860863
3: -13.1090326, 6.0949540, -13.2656479, 6.1607208, -19.2697506, 19.3606014
4: -13.0977945, 8.2355490, -13.2588930, 8.3350143, -21.4328079, 21.4944420
5: -10.0708942, 8.3360596, -10.1931314, 8.4372768, -18.5081711, 18.5291901
6: -10.7733126, 9.1170855, -10.9090014, 9.2269373, -20.0002480, 20.0260868
7: -11.6549406, 8.8459063, -11.7978649, 8.9494228, -20.6043625, 20.6437721
8: -13.4998446, 8.2280874, -13.6697006, 8.3274021, -21.8272476, 21.8977890
9: -9.6139517, 11.0848627, -9.7294683, 11.2202406, -20.8341923, 20.8143311

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
time: 4.54 seconds

## Relational analysis of NS_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
time: 3.53 seconds

## BFS NS instance: NS_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.9872684, 8.2533913, -13.2054911, 9.8948679, -20.8821373, 21.4588814
1: -8.6729527, 7.2357254, -10.4695435, 8.6546507, -17.3276024, 17.7052689
2: -14.8901701, 4.4249945, -17.8939018, 5.2840333, -20.1742039, 22.3188972
3: -12.7368670, 5.9276009, -15.3677969, 7.0606337, -19.7975006, 21.2953949
4: -12.7354136, 8.0132942, -15.3406887, 9.5820541, -22.3174667, 23.3539829
5: -9.7947311, 8.1041470, -11.7669010, 9.7368155, -19.5315475, 19.8710480
6: -10.4710789, 8.8681030, -12.5865374, 10.6396103, -21.1106892, 21.4546394
7: -11.3271027, 8.6007376, -13.6679659, 10.2880888, -21.6151924, 22.2687035
8: -13.1158285, 7.9986353, -15.7749119, 9.5331564, -22.6489849, 23.7735481
9: -9.3466797, 10.7766905, -11.2294445, 12.9544859, -22.3011627, 22.0061340

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
time: 4.88 seconds

## Relational analysis of NS_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7829011, upper bound: 16.7822746
time: 3.97 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.9850454, 9.7443323, -11.4387197, 8.5899477, -21.5749931, 21.1830521
1: -10.2908211, 8.5128374, -9.0359592, 7.5286975, -17.8195171, 17.5487976
2: -17.5809765, 5.3279200, -15.5153294, 4.6186676, -22.1996441, 20.8432503
3: -15.1083679, 6.9582453, -13.2656479, 6.1607208, -21.2690868, 20.2238922
4: -15.0804844, 9.4344416, -13.2588930, 8.3350143, -23.4154987, 22.6933327
5: -11.5762281, 9.5812635, -10.1931314, 8.4372768, -20.0135002, 19.7743950
6: -12.3843288, 10.4811668, -10.9090014, 9.2269373, -21.6112671, 21.3901672
7: -13.4824963, 10.1559601, -11.7978649, 8.9494228, -22.4319191, 21.9538250
8: -15.5203047, 9.3991623, -13.6697006, 8.3274021, -23.8477058, 23.0688629
9: -11.0642862, 12.7431612, -9.7294683, 11.2202406, -22.2845268, 22.4726295

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7836946, upper bound: 16.7824329
time: 4.03 seconds

## Relational analysis of NS_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7836946, upper bound: 16.7824329
time: 26.89 seconds

## BFS NS instance: NS_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.6844263, 9.5196877, -13.2054911, 9.8948679, -22.5792942, 22.7251778
1: -10.0478067, 8.3160458, -10.4695435, 8.6546507, -18.7024536, 18.7855892
2: -17.1670017, 5.1884336, -17.8939018, 5.2840333, -22.4510345, 23.0823326
3: -14.7486019, 6.7968574, -15.3677969, 7.0606337, -21.8092308, 22.1646538
4: -14.7311144, 9.2202826, -15.3406887, 9.5820541, -24.3131657, 24.5609703
5: -11.3113928, 9.3580503, -11.7669010, 9.7368155, -21.0482082, 21.1249504
6: -12.0951099, 10.2412014, -12.5865374, 10.6396103, -22.7347202, 22.8277397
7: -13.1684914, 9.9201641, -13.6679659, 10.2880888, -23.4565811, 23.5881310
8: -15.1530447, 9.1779852, -15.7749119, 9.5331564, -24.6861992, 24.9528961
9: -10.8052931, 12.4471617, -11.2294445, 12.9544859, -23.7597771, 23.6766052

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7836946, upper bound: 16.7824329
time: 4.90 seconds

## Relational analysis of NS_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7836946, upper bound: 16.7824329
time: 2.92 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -11.4397316, 8.5911989, -11.3226061, 8.5044737, -19.9442062, 19.9138031
1: -9.0369959, 7.5297012, -8.9438286, 7.4545755, -16.4915714, 16.4735260
2: -15.5171471, 4.6212282, -15.3532238, 4.5792804, -20.0964279, 19.9744530
3: -13.2678337, 6.1624103, -13.1354580, 6.1063895, -19.3742237, 19.2978687
4: -13.2599440, 8.3357887, -13.1238585, 8.2512798, -21.5112228, 21.4596481
5: -10.1942339, 8.4383154, -10.0906124, 8.3524094, -18.5466423, 18.5289249
6: -10.9102011, 9.2278881, -10.7946129, 9.1347942, -20.0449944, 20.0224991
7: -11.7993212, 8.9511318, -11.6781683, 8.8627644, -20.6620865, 20.6292992
8: -13.6709452, 8.3288050, -13.5267162, 8.2435799, -21.9145241, 21.8555222
9: -9.7307606, 11.2213869, -9.6327801, 11.1067209, -20.8374786, 20.8541679

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B1_A1_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822746, upper bound: 16.7829006
time: 3.71 seconds

## Relational analysis of NS_B1_A2_B1_A1_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822746, upper bound: 16.7829011
time: 3.95 seconds

## BFS NS instance: NS_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -13.2066011, 9.8960390, -11.0009003, 8.2634583, -21.4700584, 20.8969383
1: -10.4706821, 8.6556873, -8.6839771, 7.2445164, -17.7151985, 17.3396645
2: -17.8957863, 5.2866454, -14.9086437, 4.4302320, -22.3260193, 20.1952896
3: -15.3700523, 7.0623465, -12.7530680, 5.9346051, -21.3046551, 19.8154144
4: -15.3417416, 9.5830183, -12.7513933, 8.0229311, -23.3646736, 22.3344116
5: -11.7681131, 9.7378941, -9.8068142, 8.1141701, -19.8822823, 19.5447083
6: -12.5877991, 10.6405745, -10.4841347, 8.8789577, -21.4667568, 21.1247101
7: -13.6694736, 10.2898798, -11.3413324, 8.6110706, -22.2805443, 21.6312122
8: -15.7761765, 9.5345974, -13.1323118, 8.0081234, -23.7842999, 22.6669083
9: -11.2307205, 12.9558220, -9.3582220, 10.7900858, -22.0208054, 22.3140392

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822746, upper bound: 16.7829006
time: 5.04 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822746, upper bound: 16.7829011
time: 3.08 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13.1934681, 9.9035845, -10.7859278, 8.1048660, -21.2983322, 20.6895123
1: -10.4594193, 8.6505127, -8.5084677, 7.1084642, -17.5678825, 17.1589813
2: -17.8748531, 5.4263916, -14.6166058, 4.3605485, -22.2354012, 20.0429974
3: -15.3531971, 7.0678420, -12.4969711, 5.8270702, -21.1802673, 19.5648136
4: -15.3230228, 9.5862961, -12.4959555, 7.8704457, -23.1934681, 22.0822525
5: -11.7619133, 9.7373180, -9.6144400, 7.9559565, -19.7178688, 19.3517570
6: -12.5912905, 10.6488152, -10.2808628, 8.7068062, -21.2980957, 20.9296780
7: -13.6982660, 10.3232746, -11.1138897, 8.4506512, -22.1489182, 21.4371643
8: -15.7788486, 9.5592318, -12.8777771, 7.8632293, -23.6420784, 22.4370079
9: -11.2463503, 12.9484978, -9.1768284, 10.5782976, -21.8246479, 22.1253204

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of NS_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7840008, upper bound: 16.7837200
time: 29.91 seconds

## Relational analysis of NS_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7839997, upper bound: 16.7837096
time: 3.59 seconds

## BFS NS instance: NS_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -13.0089645, 9.7670755, -12.1382256, 9.1170206, -22.1259842, 21.9053001
1: -10.3103046, 8.5312157, -9.6059017, 7.9779258, -18.2882271, 18.1371174
2: -17.6231213, 5.3512421, -16.4549770, 4.9655466, -22.5886669, 21.8062191
3: -15.1329336, 6.9717898, -14.1040497, 6.5343142, -21.6672478, 21.0758400
4: -15.1074858, 9.4556236, -14.0775948, 8.8322401, -23.9397240, 23.5332184
5: -11.5989542, 9.6011715, -10.8204994, 8.9550343, -20.5539856, 20.4216709
6: -12.4158211, 10.5015583, -11.5786314, 9.7914820, -22.2073021, 22.0801888
7: -13.5053215, 10.1817970, -12.5605049, 9.5036745, -23.0089951, 22.7423019
8: -15.5565901, 9.4288054, -14.5113564, 8.8215771, -24.3781662, 23.9401627
9: -11.0891523, 12.7675791, -10.3332424, 11.9127254, -23.0018768, 23.1008186

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 97

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 240

### Candidate
type: B, layer: 1, pos: 240

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 131

### Candidate
type: B, layer: 1, pos: 131

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838215, upper bound: 16.7832483
time: 4.98 seconds

## Relational analysis of NS_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838006, upper bound: 16.7832471
time: 9.70 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 8.28 + 595.94 = 604.22 seconds
