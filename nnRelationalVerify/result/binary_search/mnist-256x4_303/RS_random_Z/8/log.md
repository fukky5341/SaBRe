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
execution time: IAR + LP analysis = 1.11 + 11.08 = 12.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227448


# Binary Search by BASE starts (time budget: 2687.80 seconds, max iter: 100)

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
Binary search time: 40.48 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2647.32 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5976364, upper bound: 264.5976364
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5976364, upper bound: 264.5976364
time: 6.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.06 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 13.06
Output dim: 7, lower bound: -264.5976364, upper bound: 264.5976364
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 13.06
Output dim: 7, lower bound: -264.5976364, upper bound: 264.5976364
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=266.34649658203125
rel_dist={7: [-264.62269597868163, 264.6226959786816]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6091311, upper bound: 264.6091309
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6091311, upper bound: 264.6091309
time: 7.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.13 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 15.13
Output dim: 7, lower bound: -264.6091311, upper bound: 264.6091309
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 15.13
Output dim: 7, lower bound: -264.6091311, upper bound: 264.6091309
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=266.34649658203125
rel_dist={7: [-264.6227209801136, 264.62272095543653]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6223272, upper bound: 264.6223270
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6223270, upper bound: 264.6223272
time: 7.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.38
Output dim: 7, lower bound: -264.6223272, upper bound: 264.6223270
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.38
Output dim: 7, lower bound: -264.6223270, upper bound: 264.6223272

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

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6072945, upper bound: 264.6072943
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6072945, upper bound: 264.6072943
time: 7.18 seconds

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

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5958844, upper bound: 264.5958891
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.5958844, upper bound: 264.5958891
time: 6.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.98 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.98
Output dim: 7, lower bound: -264.6072945, upper bound: 264.6072943
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.98
Output dim: 7, lower bound: -264.6072945, upper bound: 264.6072943
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.98
Output dim: 7, lower bound: -264.5958844, upper bound: 264.5958891
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.98
Output dim: 7, lower bound: -264.5958844, upper bound: 264.5958891
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=266.34649658203125
rel_dist={7: [-264.6227371509656, 264.6227371509656]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6091798, upper bound: 264.6091796
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6091798, upper bound: 264.6091796
time: 6.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.58 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 13.58
Output dim: 7, lower bound: -264.6091798, upper bound: 264.6091796
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 13.58
Output dim: 7, lower bound: -264.6091798, upper bound: 264.6091796
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=266.34649658203125
rel_dist={7: [-264.6227447629983, 264.6227447629983]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 135.05 seconds
