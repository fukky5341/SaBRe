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
execution time: IAR + RelationalAnalysis = 2.06 + 6.04 = 8.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -16.7889211, upper bound: 16.7889210

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888423
time: 3.79 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.43 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.43
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.43
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888423

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882387
time: 5.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882385
time: 13.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882385
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882385
time: 3.41 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.24 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.24
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882387
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.24
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882385
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.24
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882385
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.24
Output dim: 2, lower bound: -16.7882387, upper bound: 16.7882385

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864457
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864458
time: 2.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864453
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864458
time: 2.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864459
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864459
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864455
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864455
time: 3.21 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 8.61 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864457
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864458
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864453
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864458
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864459
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864459
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864455
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.61
Output dim: 2, lower bound: -16.7864459, upper bound: 16.7864455

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857602
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857612
time: 4.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857602
time: 3.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857606
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857603
time: 3.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857606
time: 14.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857612
time: 29.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857612
time: 37.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857607
time: 26.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857603
time: 11.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857603
time: 28.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
time: 11.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857601
time: 11.58 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857602
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857612
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857602
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857606
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857603
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857606
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857612
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857612
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857607
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857603
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857603
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857608
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.49
Output dim: 2, lower bound: -16.7857612, upper bound: 16.7857601

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 2.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 3.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
time: 2.66 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 9.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 9.40
Output dim: 2, lower bound: -16.7850241, upper bound: 16.7850241

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 7.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 4.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 5.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 3.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 3.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7828460, upper bound: 16.7828460
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205
1: -11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597
2: -20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066
3: -17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639
4: -17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085
5: -13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499
6: -14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247
7: -15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786
8: -18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871
9: -12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 184

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 8.10 + 593.87 = 601.97 seconds
