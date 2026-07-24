## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 264.612307261
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660)
1: (-120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653)
2: (-158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565)
3: (-169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952)
4: (-154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305)
5: (-139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240)
6: (-133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709)
7: (-144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966)
8: (-174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793)
9: (-131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962)

## BASE Result
execution time: IAR + LP analysis = 1.09 + 11.10 = 12.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227448


# Binary Search by BASE starts (time budget: 2687.82 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=266.34649658203125
rel_dist={7: [-264.62269597868163, 264.6226959786816]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=266.34649658203125
rel_dist={7: [-264.62266420921173, 264.6226641973501]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=266.34649658203125
rel_dist={7: [-264.6222950859965, 264.6222950868074]}

## Binary Search Result
Binary search time: 40.62 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2647.20 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6226960, upper bound: 264.6226952
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6226952, upper bound: 264.6226960
time: 6.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.52
Output dim: 7, lower bound: -264.6226960, upper bound: 264.6226952
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.52
Output dim: 7, lower bound: -264.6226952, upper bound: 264.6226960

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212889, upper bound: 264.6212428
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212615
time: 7.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212615, upper bound: 264.6212575
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212428, upper bound: 264.6212889
time: 7.26 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -264.6212889, upper bound: 264.6212428
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212615
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -264.6212615, upper bound: 264.6212575
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.01
Output dim: 7, lower bound: -264.6212428, upper bound: 264.6212889

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203032, upper bound: 264.6202748
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6202950, upper bound: 264.6202785
time: 7.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6202803, upper bound: 264.6202930
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6202754, upper bound: 264.6202971
time: 7.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6202971, upper bound: 264.6202754
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6202930, upper bound: 264.6202803
time: 7.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6202785, upper bound: 264.6202950
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6202748, upper bound: 264.6203032
time: 8.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6203032, upper bound: 264.6202748
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6202950, upper bound: 264.6202785
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6202803, upper bound: 264.6202930
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6202754, upper bound: 264.6202971
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6202971, upper bound: 264.6202754
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6202930, upper bound: 264.6202803
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6202785, upper bound: 264.6202950
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.81
Output dim: 7, lower bound: -264.6202748, upper bound: 264.6203032

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192345, upper bound: 264.6192041
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192145, upper bound: 264.6192228
time: 7.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192323, upper bound: 264.6192048
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192131, upper bound: 264.6192273
time: 8.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192277, upper bound: 264.6192115
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192064, upper bound: 264.6192323
time: 8.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192229, upper bound: 264.6192125
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192045, upper bound: 264.6192336
time: 7.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192336, upper bound: 264.6192045
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192125, upper bound: 264.6192229
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192323, upper bound: 264.6192064
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192115, upper bound: 264.6192277
time: 7.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192273, upper bound: 264.6192131
time: 9.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192048, upper bound: 264.6192323
time: 8.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192228, upper bound: 264.6192145
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192041, upper bound: 264.6192345
time: 7.05 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192345, upper bound: 264.6192041
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192145, upper bound: 264.6192228
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192323, upper bound: 264.6192048
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192131, upper bound: 264.6192273
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192277, upper bound: 264.6192115
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192064, upper bound: 264.6192323
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192229, upper bound: 264.6192125
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192045, upper bound: 264.6192336
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192336, upper bound: 264.6192045
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192125, upper bound: 264.6192229
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192323, upper bound: 264.6192064
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192115, upper bound: 264.6192277
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192273, upper bound: 264.6192131
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192048, upper bound: 264.6192323
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192228, upper bound: 264.6192145
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.54
Output dim: 7, lower bound: -264.6192041, upper bound: 264.6192345

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
time: 7.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
time: 7.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
time: 8.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
time: 8.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
time: 7.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
time: 8.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
time: 8.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
time: 7.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
time: 7.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
time: 7.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
time: 7.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
time: 8.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
time: 7.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
time: 8.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
time: 6.69 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049346, upper bound: 264.6049166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049178, upper bound: 264.6049277
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049207, upper bound: 264.6049190
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049144, upper bound: 264.6049398
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049398, upper bound: 264.6049144
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049190, upper bound: 264.6049207
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049277, upper bound: 264.6049178
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.53
Output dim: 7, lower bound: -264.6049166, upper bound: 264.6049346
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=266.34649658203125
rel_dist={7: [-264.62269597868163, 264.6226959786816]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227210, upper bound: 264.6227201
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227201, upper bound: 264.6227210
time: 7.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.86
Output dim: 7, lower bound: -264.6227210, upper bound: 264.6227201
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.86
Output dim: 7, lower bound: -264.6227201, upper bound: 264.6227210

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213327, upper bound: 264.6212777
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212976, upper bound: 264.6213002
time: 6.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213002, upper bound: 264.6212976
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212777, upper bound: 264.6213327
time: 7.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.58
Output dim: 7, lower bound: -264.6213327, upper bound: 264.6212777
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.58
Output dim: 7, lower bound: -264.6212976, upper bound: 264.6213002
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.58
Output dim: 7, lower bound: -264.6213002, upper bound: 264.6212976
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.58
Output dim: 7, lower bound: -264.6212777, upper bound: 264.6213327

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203433, upper bound: 264.6203083
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203351, upper bound: 264.6203129
time: 8.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203138, upper bound: 264.6203328
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203086, upper bound: 264.6203394
time: 7.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203394, upper bound: 264.6203086
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203328, upper bound: 264.6203138
time: 7.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203129, upper bound: 264.6203351
time: 7.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203083, upper bound: 264.6203433
time: 7.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203433, upper bound: 264.6203083
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203351, upper bound: 264.6203129
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203138, upper bound: 264.6203328
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203086, upper bound: 264.6203394
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203394, upper bound: 264.6203086
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203328, upper bound: 264.6203138
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203129, upper bound: 264.6203351
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.09
Output dim: 7, lower bound: -264.6203083, upper bound: 264.6203433

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192725, upper bound: 264.6192294
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192447, upper bound: 264.6192546
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192693, upper bound: 264.6192310
time: 9.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192424, upper bound: 264.6192613
time: 8.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192624, upper bound: 264.6192410
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192333, upper bound: 264.6192693
time: 7.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192550, upper bound: 264.6192434
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192303, upper bound: 264.6192723
time: 7.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192723, upper bound: 264.6192303
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192434, upper bound: 264.6192551
time: 7.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192693, upper bound: 264.6192333
time: 8.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192410, upper bound: 264.6192624
time: 7.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192613, upper bound: 264.6192424
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192310, upper bound: 264.6192693
time: 8.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192546, upper bound: 264.6192447
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192294, upper bound: 264.6192725
time: 7.26 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192725, upper bound: 264.6192294
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192447, upper bound: 264.6192546
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192693, upper bound: 264.6192310
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192424, upper bound: 264.6192613
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192624, upper bound: 264.6192410
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192333, upper bound: 264.6192693
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192550, upper bound: 264.6192434
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192303, upper bound: 264.6192723
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192723, upper bound: 264.6192303
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192434, upper bound: 264.6192551
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192693, upper bound: 264.6192333
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192410, upper bound: 264.6192624
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192613, upper bound: 264.6192424
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192310, upper bound: 264.6192693
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192546, upper bound: 264.6192447
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.71
Output dim: 7, lower bound: -264.6192294, upper bound: 264.6192725

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
time: 7.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
time: 6.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
time: 7.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
time: 6.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
time: 6.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
time: 28.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
time: 6.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
time: 7.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049519
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049895, upper bound: 264.6049519
time: 7.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
time: 7.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
time: 7.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
time: 8.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
time: 7.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049895, upper bound: 264.6049519
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049895, upper bound: 264.6049519
time: 6.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
time: 7.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
time: 8.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
time: 7.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049564, upper bound: 264.6049717
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049623, upper bound: 264.6049587
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049519, upper bound: 264.6049895
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049519
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049895, upper bound: 264.6049519
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049895, upper bound: 264.6049519
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049895, upper bound: 264.6049519
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049587, upper bound: 264.6049623
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049717, upper bound: 264.6049564
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 15.86
Output dim: 7, lower bound: -264.6049548, upper bound: 264.6049830
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=266.34649658203125
rel_dist={7: [-264.6227209801136, 264.62272095543653]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227372, upper bound: 264.6227357
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227357, upper bound: 264.6227372
time: 7.47 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.81
Output dim: 7, lower bound: -264.6227372, upper bound: 264.6227357
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.81
Output dim: 7, lower bound: -264.6227357, upper bound: 264.6227372

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213584, upper bound: 264.6212998
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213233, upper bound: 264.6213243
time: 7.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213243, upper bound: 264.6213233
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212998, upper bound: 264.6213584
time: 8.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.57
Output dim: 7, lower bound: -264.6213584, upper bound: 264.6212998
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.57
Output dim: 7, lower bound: -264.6213233, upper bound: 264.6213243
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.57
Output dim: 7, lower bound: -264.6213243, upper bound: 264.6213233
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.57
Output dim: 7, lower bound: -264.6212998, upper bound: 264.6213584

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203678, upper bound: 264.6203292
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203598, upper bound: 264.6203342
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203349, upper bound: 264.6203570
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203295, upper bound: 264.6203647
time: 7.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203647, upper bound: 264.6203295
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203570, upper bound: 264.6203349
time: 7.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203342, upper bound: 264.6203598
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203292, upper bound: 264.6203678
time: 8.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203678, upper bound: 264.6203292
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203598, upper bound: 264.6203342
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203349, upper bound: 264.6203570
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203295, upper bound: 264.6203647
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203647, upper bound: 264.6203295
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203570, upper bound: 264.6203349
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203342, upper bound: 264.6203598
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.63
Output dim: 7, lower bound: -264.6203292, upper bound: 264.6203678

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192924, upper bound: 264.6192444
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192637, upper bound: 264.6192753
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192924, upper bound: 264.6192472
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192596, upper bound: 264.6192824
time: 7.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192832, upper bound: 264.6192579
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192505, upper bound: 264.6192924
time: 8.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192759, upper bound: 264.6192611
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192456, upper bound: 264.6192959
time: 7.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192959, upper bound: 264.6192456
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192611, upper bound: 264.6192759
time: 8.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192924, upper bound: 264.6192505
time: 10.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192579, upper bound: 264.6192832
time: 7.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192824, upper bound: 264.6192596
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192472, upper bound: 264.6192924
time: 6.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192753, upper bound: 264.6192637
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192444, upper bound: 264.6192960
time: 7.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192924, upper bound: 264.6192444
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192637, upper bound: 264.6192753
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192924, upper bound: 264.6192472
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192596, upper bound: 264.6192824
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192832, upper bound: 264.6192579
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192505, upper bound: 264.6192924
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192759, upper bound: 264.6192611
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192456, upper bound: 264.6192959
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192959, upper bound: 264.6192456
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192611, upper bound: 264.6192759
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192924, upper bound: 264.6192505
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192579, upper bound: 264.6192832
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192824, upper bound: 264.6192596
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192472, upper bound: 264.6192924
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192753, upper bound: 264.6192637
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.15
Output dim: 7, lower bound: -264.6192444, upper bound: 264.6192960

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
time: 7.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
time: 7.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
time: 7.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
time: 7.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
time: 6.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
time: 7.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
time: 6.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
time: 6.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
time: 6.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
time: 6.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
time: 7.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
time: 6.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
time: 7.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
time: 7.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
time: 7.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
time: 7.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050137, upper bound: 264.6049773
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049800, upper bound: 264.6049996
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049894, upper bound: 264.6049830
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049744, upper bound: 264.6050201
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6050201, upper bound: 264.6049744
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049830, upper bound: 264.6049894
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049996, upper bound: 264.6049800
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.45
Output dim: 7, lower bound: -264.6049773, upper bound: 264.6050137
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=266.34649658203125
rel_dist={7: [-264.6227371509656, 264.6227371509656]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227431
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6227431, upper bound: 264.6227448
time: 6.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.49
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227431
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.49
Output dim: 7, lower bound: -264.6227431, upper bound: 264.6227448

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213695, upper bound: 264.6213105
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213342, upper bound: 264.6213359
time: 7.54 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213359, upper bound: 264.6213342
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213105, upper bound: 264.6213695
time: 7.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.84
Output dim: 7, lower bound: -264.6213695, upper bound: 264.6213105
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.84
Output dim: 7, lower bound: -264.6213342, upper bound: 264.6213359
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.84
Output dim: 7, lower bound: -264.6213359, upper bound: 264.6213342
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.84
Output dim: 7, lower bound: -264.6213105, upper bound: 264.6213695

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203789, upper bound: 264.6203392
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203716, upper bound: 264.6203445
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203451, upper bound: 264.6203683
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203396, upper bound: 264.6203760
time: 7.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203760, upper bound: 264.6203396
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203683, upper bound: 264.6203450
time: 8.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203445, upper bound: 264.6203716
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6203392, upper bound: 264.6203789
time: 7.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203789, upper bound: 264.6203392
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203716, upper bound: 264.6203445
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203451, upper bound: 264.6203683
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203396, upper bound: 264.6203760
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203760, upper bound: 264.6203396
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203683, upper bound: 264.6203450
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203445, upper bound: 264.6203716
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.23
Output dim: 7, lower bound: -264.6203392, upper bound: 264.6203789

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6193061, upper bound: 264.6192518
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192730, upper bound: 264.6192850
time: 7.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6193030, upper bound: 264.6192552
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192680, upper bound: 264.6192926
time: 7.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192930, upper bound: 264.6192659
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192588, upper bound: 264.6193029
time: 7.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192532, upper bound: 264.6192694
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192532, upper bound: 264.6193058
time: 7.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6193058, upper bound: 264.6192532
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192694, upper bound: 264.6192858
time: 7.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6193029, upper bound: 264.6192588
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192659, upper bound: 264.6192930
time: 8.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192926, upper bound: 264.6192680
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192552, upper bound: 264.6193030
time: 6.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192850, upper bound: 264.6192730
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6192518, upper bound: 264.6193061
time: 7.48 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6193061, upper bound: 264.6192518
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192730, upper bound: 264.6192850
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6193030, upper bound: 264.6192552
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192680, upper bound: 264.6192926
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192930, upper bound: 264.6192659
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192588, upper bound: 264.6193029
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192532, upper bound: 264.6192694
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192532, upper bound: 264.6193058
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6193058, upper bound: 264.6192532
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192694, upper bound: 264.6192858
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6193029, upper bound: 264.6192588
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192659, upper bound: 264.6192930
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192926, upper bound: 264.6192680
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192552, upper bound: 264.6193030
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192850, upper bound: 264.6192730
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.31
Output dim: 7, lower bound: -264.6192518, upper bound: 264.6193061

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
time: 6.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049945
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049945
time: 7.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
time: 6.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
time: 6.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049944
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049944
time: 6.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
time: 7.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
time: 6.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
time: 7.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
time: 6.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
time: 6.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
time: 6.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
time: 6.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660
1: -120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653
2: -158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565
3: -169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952
4: -154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305
5: -139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240
6: -133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709
7: -144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966
8: -174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793
9: -131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
time: 6.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049945
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049945
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050282, upper bound: 264.6049879
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049910, upper bound: 264.6050133
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049944
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050028, upper bound: 264.6049944
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049852, upper bound: 264.6050346
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050346, upper bound: 264.6049852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049945, upper bound: 264.6050028
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6050133, upper bound: 264.6049910
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.40
Output dim: 7, lower bound: -264.6049879, upper bound: 264.6050282
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=266.34649658203125
rel_dist={7: [-264.6227447629983, 264.6227447629983]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 2052.68 seconds
