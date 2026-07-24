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
execution time: IAR + RelationalAnalysis = 0.87 + 5.94 = 6.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -16.7889211, upper bound: 16.7889210

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7889210, upper bound: 16.7889208
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7889211, upper bound: 16.7889210
time: 2.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.61 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.61
Output dim: 2, lower bound: -16.7889210, upper bound: 16.7889208
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.61
Output dim: 2, lower bound: -16.7889211, upper bound: 16.7889210

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888900, upper bound: 16.7888897
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888900, upper bound: 16.7888900
time: 5.20 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422
time: 3.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.79
Output dim: 2, lower bound: -16.7888900, upper bound: 16.7888897
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.79
Output dim: 2, lower bound: -16.7888900, upper bound: 16.7888900
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.79
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.79
Output dim: 2, lower bound: -16.7888424, upper bound: 16.7888422

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888899, upper bound: 16.7888900
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7888900, upper bound: 16.7888897
time: 3.43 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866587, upper bound: 16.7866587
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866587, upper bound: 16.7866587
time: 2.05 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883654, upper bound: 16.7883650
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883654, upper bound: 16.7883650
time: 4.77 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7885273, upper bound: 16.7885267
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7885273, upper bound: 16.7885267
time: 4.57 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 9.27 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7888899, upper bound: 16.7888900
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7888900, upper bound: 16.7888897
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7866587, upper bound: 16.7866587
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7866587, upper bound: 16.7866587
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7883654, upper bound: 16.7883650
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7883654, upper bound: 16.7883650
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7885273, upper bound: 16.7885267
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7885273, upper bound: 16.7885267

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877293, upper bound: 16.7877298
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877293, upper bound: 16.7877298
time: 3.28 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7882797, upper bound: 16.7882801
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7882797, upper bound: 16.7882801
time: 3.97 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863881
time: 2.18 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 160

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866255
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866255
time: 2.31 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883650, upper bound: 16.7883650
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7883654, upper bound: 16.7883646
time: 2.81 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877572
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877570
time: 3.31 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856130, upper bound: 16.7856130
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856130, upper bound: 16.7856125
time: 2.94 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7885273, upper bound: 16.7885266
time: 7.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7885268, upper bound: 16.7885267
time: 6.34 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7877293, upper bound: 16.7877298
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7877293, upper bound: 16.7877298
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7882797, upper bound: 16.7882801
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7882797, upper bound: 16.7882801
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863881
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866255
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866255
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7883650, upper bound: 16.7883650
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7883654, upper bound: 16.7883646
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877572
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877570
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7856130, upper bound: 16.7856130
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7856130, upper bound: 16.7856125
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7885273, upper bound: 16.7885266
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.54
Output dim: 2, lower bound: -16.7885268, upper bound: 16.7885267

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856135, upper bound: 16.7856131
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856135, upper bound: 16.7856128
time: 2.42 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877287, upper bound: 16.7877300
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877293, upper bound: 16.7877290
time: 4.02 seconds

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858542, upper bound: 16.7858545
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858542, upper bound: 16.7858544
time: 2.97 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876263
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876264
time: 10.47 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
time: 2.56 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861508
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861505
time: 4.00 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 255

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866256, upper bound: 16.7866257
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866254
time: 2.44 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866254, upper bound: 16.7866257
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866254
time: 3.09 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878496, upper bound: 16.7878507
time: 2.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878508, upper bound: 16.7878497
time: 2.98 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877570
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877570
time: 2.92 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7871395, upper bound: 16.7871396
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7871395, upper bound: 16.7871395
time: 2.75 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877486
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877486
time: 2.76 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838838, upper bound: 16.7838837
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838838, upper bound: 16.7838837
time: 2.97 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856127, upper bound: 16.7856119
time: 6.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856130, upper bound: 16.7856127
time: 3.63 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 158

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857275
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857274
time: 3.59 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 158

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857275
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857275
time: 2.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7856135, upper bound: 16.7856131
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7856135, upper bound: 16.7856128
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7877287, upper bound: 16.7877300
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7877293, upper bound: 16.7877290
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7858542, upper bound: 16.7858545
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7858542, upper bound: 16.7858544
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876263
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876264
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861508
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861505
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7866256, upper bound: 16.7866257
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866254
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7866254, upper bound: 16.7866257
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7866257, upper bound: 16.7866254
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7878496, upper bound: 16.7878507
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7878508, upper bound: 16.7878497
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877570
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7877573, upper bound: 16.7877570
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7871395, upper bound: 16.7871396
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7871395, upper bound: 16.7871395
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877486
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877486
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7838838, upper bound: 16.7838837
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7838838, upper bound: 16.7838837
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7856127, upper bound: 16.7856119
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7856130, upper bound: 16.7856127
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857274
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857275
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857275

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854824
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854824
time: 3.60 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848297
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848298
time: 4.84 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7869501, upper bound: 16.7869498
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7869501, upper bound: 16.7869499
time: 3.93 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876718, upper bound: 16.7876720
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876718, upper bound: 16.7876719
time: 3.19 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 158

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846111, upper bound: 16.7846102
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846111, upper bound: 16.7846104
time: 2.68 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858541, upper bound: 16.7858544
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858542, upper bound: 16.7858543
time: 4.48 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876257, upper bound: 16.7876255
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876258, upper bound: 16.7876252
time: 2.47 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 158

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876264
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876261
time: 3.44 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863881
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
time: 2.10 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7847413, upper bound: 16.7847414
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7847413, upper bound: 16.7847414
time: 2.37 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7861507, upper bound: 16.7861505
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861503
time: 3.09 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7861506, upper bound: 16.7861505
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861502
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7851196, upper bound: 16.7851195
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7851196, upper bound: 16.7851195
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7837780, upper bound: 16.7837780
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7837780, upper bound: 16.7837774
time: 2.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 160

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7851225, upper bound: 16.7851224
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7851225, upper bound: 16.7851217
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7837782, upper bound: 16.7837770
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7837782, upper bound: 16.7837777
time: 2.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878323, upper bound: 16.7878339
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878323, upper bound: 16.7878338
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 74

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875409, upper bound: 16.7875403
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875409, upper bound: 16.7875404
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7845554, upper bound: 16.7845550
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7845554, upper bound: 16.7845550
time: 2.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855738, upper bound: 16.7855731
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855738, upper bound: 16.7855731
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7867232, upper bound: 16.7867229
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7867232, upper bound: 16.7867230
time: 3.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822149, upper bound: 16.7822145
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7822149, upper bound: 16.7822146
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877488
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877485
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877477, upper bound: 16.7877478
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877479, upper bound: 16.7877478
time: 4.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838838, upper bound: 16.7838834
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838606, upper bound: 16.7838605
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838606, upper bound: 16.7838605
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855119, upper bound: 16.7855109
time: 6.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7855119, upper bound: 16.7855109
time: 2.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857278
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857280, upper bound: 16.7857280
time: 1.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857281, upper bound: 16.7857275
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857278
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856512, upper bound: 16.7856503
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7856512, upper bound: 16.7856514
time: 5.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7844637, upper bound: 16.7844636
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7844637, upper bound: 16.7844636
time: 1.73 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854824
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854824
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848297
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848298
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7869501, upper bound: 16.7869498
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7869501, upper bound: 16.7869499
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7876718, upper bound: 16.7876720
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7876718, upper bound: 16.7876719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7846111, upper bound: 16.7846102
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7846111, upper bound: 16.7846104
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7858541, upper bound: 16.7858544
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7858542, upper bound: 16.7858543
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7876257, upper bound: 16.7876255
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7876258, upper bound: 16.7876252
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876264
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876261
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863881
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7847413, upper bound: 16.7847414
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7847413, upper bound: 16.7847414
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7861507, upper bound: 16.7861505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861503
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7861506, upper bound: 16.7861505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861502
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7851196, upper bound: 16.7851195
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7851196, upper bound: 16.7851195
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7837780, upper bound: 16.7837780
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7837780, upper bound: 16.7837774
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7851225, upper bound: 16.7851224
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7851225, upper bound: 16.7851217
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7837782, upper bound: 16.7837770
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7837782, upper bound: 16.7837777
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7878323, upper bound: 16.7878339
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7878323, upper bound: 16.7878338
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7875409, upper bound: 16.7875403
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7875409, upper bound: 16.7875404
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7845554, upper bound: 16.7845550
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7845554, upper bound: 16.7845550
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7855738, upper bound: 16.7855731
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7855738, upper bound: 16.7855731
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7867232, upper bound: 16.7867229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7867232, upper bound: 16.7867230
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7822149, upper bound: 16.7822145
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7822149, upper bound: 16.7822146
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877488
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877485
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7877477, upper bound: 16.7877478
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7877479, upper bound: 16.7877478
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7838838, upper bound: 16.7838834
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7838606, upper bound: 16.7838605
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7838606, upper bound: 16.7838605
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7855119, upper bound: 16.7855109
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7855119, upper bound: 16.7855109
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7857280, upper bound: 16.7857280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7857281, upper bound: 16.7857275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7856512, upper bound: 16.7856503
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7856512, upper bound: 16.7856514
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7844637, upper bound: 16.7844636
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 2, lower bound: -16.7844637, upper bound: 16.7844636

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854823
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854823
time: 3.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 158

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7839574, upper bound: 16.7839572
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7839574, upper bound: 16.7839573
time: 3.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848297
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848296
time: 3.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7843650, upper bound: 16.7843647
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7843650, upper bound: 16.7843646
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7859874, upper bound: 16.7859871
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7859874, upper bound: 16.7859878
time: 4.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7869454, upper bound: 16.7869451
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7869454, upper bound: 16.7869454
time: 13.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874040, upper bound: 16.7874038
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874040, upper bound: 16.7874038
time: 8.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7862742, upper bound: 16.7862728
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7862742, upper bound: 16.7862728
time: 2.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 74

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7827289, upper bound: 16.7827281
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7827289, upper bound: 16.7827287
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846098, upper bound: 16.7846096
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7846098, upper bound: 16.7846088
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858537, upper bound: 16.7858544
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858541, upper bound: 16.7858539
time: 4.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858473, upper bound: 16.7858476
time: 6.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7858475, upper bound: 16.7858472
time: 2.44 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 9.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854823
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7854825, upper bound: 16.7854823
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7839574, upper bound: 16.7839572
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7839574, upper bound: 16.7839573
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848297
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7848299, upper bound: 16.7848296
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7843650, upper bound: 16.7843647
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7843650, upper bound: 16.7843646
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7859874, upper bound: 16.7859871
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7859874, upper bound: 16.7859878
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7869454, upper bound: 16.7869451
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7869454, upper bound: 16.7869454
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7874040, upper bound: 16.7874038
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7874040, upper bound: 16.7874038
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7862742, upper bound: 16.7862728
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7862742, upper bound: 16.7862728
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7827289, upper bound: 16.7827281
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7827289, upper bound: 16.7827287
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7846098, upper bound: 16.7846096
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7846098, upper bound: 16.7846088
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7858537, upper bound: 16.7858544
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7858541, upper bound: 16.7858539
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7858473, upper bound: 16.7858476
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 9.38
Output dim: 2, lower bound: -16.7858475, upper bound: 16.7858472
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7876257, upper bound: 16.7876255
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7876258, upper bound: 16.7876252
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876264
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7876265, upper bound: 16.7876261
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863881
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7863884, upper bound: 16.7863884
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7847413, upper bound: 16.7847414
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7847413, upper bound: 16.7847414
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7861507, upper bound: 16.7861505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861503
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7861506, upper bound: 16.7861505
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7861508, upper bound: 16.7861502
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7851196, upper bound: 16.7851195
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7851196, upper bound: 16.7851195
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7837780, upper bound: 16.7837780
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7837780, upper bound: 16.7837774
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7851225, upper bound: 16.7851224
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7851225, upper bound: 16.7851217
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7837782, upper bound: 16.7837770
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7837782, upper bound: 16.7837777
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7878323, upper bound: 16.7878339
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7878323, upper bound: 16.7878338
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7875409, upper bound: 16.7875403
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7875409, upper bound: 16.7875404
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7845554, upper bound: 16.7845550
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7845554, upper bound: 16.7845550
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7855738, upper bound: 16.7855731
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7855738, upper bound: 16.7855731
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7867232, upper bound: 16.7867229
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7867232, upper bound: 16.7867230
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7822149, upper bound: 16.7822145
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7822149, upper bound: 16.7822146
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877488
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7877487, upper bound: 16.7877485
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7877477, upper bound: 16.7877478
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7877479, upper bound: 16.7877478
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7838838, upper bound: 16.7838834
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7838606, upper bound: 16.7838605
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7838606, upper bound: 16.7838605
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7838834, upper bound: 16.7838838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7855119, upper bound: 16.7855109
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7855119, upper bound: 16.7855109
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7857280, upper bound: 16.7857280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7857281, upper bound: 16.7857275
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7857283, upper bound: 16.7857278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7856512, upper bound: 16.7856503
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7856512, upper bound: 16.7856514
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7844637, upper bound: 16.7844636
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 9.38
Output dim: 2, lower bound: -16.7844637, upper bound: 16.7844636

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 6.81 + 599.23 = 606.04 seconds
