## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 143.61867486269998
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766)
1: (-62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901)
2: (-83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913)
3: (-88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352)
4: (-81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725)
5: (-73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507)
6: (-70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321)
7: (-76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043)
8: (-92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513)
9: (-69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561)

## BASE Result
execution time: IAR + LP analysis = 1.27 + 8.40 = 9.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624614, upper bound: 143.7624614


# Binary Search by BASE starts (time budget: 1990.33 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0042724609375
rel_dist={4: [-143.76245312009848, 143.76245312009854]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0042724609375
rel_dist={4: [-143.76243730833275, 143.76243730833278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0042724609375
rel_dist={4: [-143.7624133928287, 143.76241339486478]}

## Binary Search Result
Binary search time: 33.81 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1956.52 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7602417, upper bound: 143.7602415
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7602415, upper bound: 143.7602417
time: 5.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.14 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.14
Output dim: 4, lower bound: -143.7602417, upper bound: 143.7602415
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.14
Output dim: 4, lower bound: -143.7602415, upper bound: 143.7602417

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600510, upper bound: 143.7600424
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600434, upper bound: 143.7600512
time: 5.90 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600512, upper bound: 143.7600434
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600424, upper bound: 143.7600510
time: 5.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 11.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.95
Output dim: 4, lower bound: -143.7600510, upper bound: 143.7600424
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.95
Output dim: 4, lower bound: -143.7600434, upper bound: 143.7600512
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.95
Output dim: 4, lower bound: -143.7600512, upper bound: 143.7600434
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.95
Output dim: 4, lower bound: -143.7600424, upper bound: 143.7600510

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7582031, upper bound: 143.7581818
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581989, upper bound: 143.7581826
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581854, upper bound: 143.7581989
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581838, upper bound: 143.7582023
time: 5.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7582023, upper bound: 143.7581838
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581989, upper bound: 143.7581854
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581826, upper bound: 143.7581989
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581818, upper bound: 143.7582031
time: 5.48 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7582031, upper bound: 143.7581818
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7581989, upper bound: 143.7581826
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7581854, upper bound: 143.7581989
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7581838, upper bound: 143.7582023
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7582023, upper bound: 143.7581838
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7581989, upper bound: 143.7581854
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7581826, upper bound: 143.7581989
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.93
Output dim: 4, lower bound: -143.7581818, upper bound: 143.7582031

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579660, upper bound: 143.7579103
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579583, upper bound: 143.7579173
time: 5.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579576, upper bound: 143.7579125
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579475, upper bound: 143.7579186
time: 6.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579231, upper bound: 143.7579447
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579179, upper bound: 143.7579558
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579209, upper bound: 143.7579541
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579179, upper bound: 143.7579644
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579644, upper bound: 143.7579179
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579541, upper bound: 143.7579209
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579558, upper bound: 143.7579222
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579447, upper bound: 143.7579231
time: 6.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579186, upper bound: 143.7579475
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579125, upper bound: 143.7579576
time: 6.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579173, upper bound: 143.7579583
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579103, upper bound: 143.7579660
time: 5.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579660, upper bound: 143.7579103
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579583, upper bound: 143.7579173
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579576, upper bound: 143.7579125
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579475, upper bound: 143.7579186
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579231, upper bound: 143.7579447
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579179, upper bound: 143.7579558
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579209, upper bound: 143.7579541
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579179, upper bound: 143.7579644
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579644, upper bound: 143.7579179
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579541, upper bound: 143.7579209
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579558, upper bound: 143.7579222
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579447, upper bound: 143.7579231
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579186, upper bound: 143.7579475
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579125, upper bound: 143.7579576
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579173, upper bound: 143.7579583
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.73
Output dim: 4, lower bound: -143.7579103, upper bound: 143.7579660

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
time: 4.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
time: 5.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 11.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274973
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275294
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.51
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
time: 4.51 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 10.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275321, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275095, upper bound: 143.6274847
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275207, upper bound: 143.6274945
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274995, upper bound: 143.6274973
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275075, upper bound: 143.6274969
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275031, upper bound: 143.6275179
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274877, upper bound: 143.6275049
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6274865, upper bound: 143.6275294
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275294, upper bound: 143.6274865
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.47
Output dim: 4, lower bound: -143.6275049, upper bound: 143.6274877
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6275179, upper bound: 143.6275075
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274973, upper bound: 143.6275207
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.47
Output dim: 4, lower bound: -143.6274847, upper bound: 143.6275321
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0042724609375
rel_dist={4: [-143.76245312009848, 143.76245312009854]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 5.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.80
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.80
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600166
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600175, upper bound: 143.7600212
time: 6.17 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600175
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600166, upper bound: 143.7600212
time: 5.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.98
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600166
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.98
Output dim: 4, lower bound: -143.7600175, upper bound: 143.7600212
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.98
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600175
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.98
Output dim: 4, lower bound: -143.7600166, upper bound: 143.7600212

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581578, upper bound: 143.7581483
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581483
time: 5.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581495, upper bound: 143.7581576
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581485, upper bound: 143.7581577
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581577, upper bound: 143.7581485
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581495
time: 6.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581576
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581578
time: 5.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.56 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581578, upper bound: 143.7581483
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581483
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581495, upper bound: 143.7581576
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581485, upper bound: 143.7581577
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581577, upper bound: 143.7581485
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581495
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581576
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.56
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581578

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579293, upper bound: 143.7579012
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579255, upper bound: 143.7579048
time: 6.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579260, upper bound: 143.7579025
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579207, upper bound: 143.7579055
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579078, upper bound: 143.7579188
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579071, upper bound: 143.7579250
time: 6.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579068, upper bound: 143.7579234
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579049, upper bound: 143.7579290
time: 7.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579290, upper bound: 143.7579049
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579234, upper bound: 143.7579068
time: 5.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579250, upper bound: 143.7579071
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579188, upper bound: 143.7579078
time: 6.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579055, upper bound: 143.7579207
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579025, upper bound: 143.7579260
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579048, upper bound: 143.7579255
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579012, upper bound: 143.7579294
time: 6.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579293, upper bound: 143.7579012
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579255, upper bound: 143.7579048
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579260, upper bound: 143.7579025
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579207, upper bound: 143.7579055
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579078, upper bound: 143.7579188
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579071, upper bound: 143.7579250
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579068, upper bound: 143.7579234
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579049, upper bound: 143.7579290
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579290, upper bound: 143.7579049
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579234, upper bound: 143.7579068
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579250, upper bound: 143.7579071
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579188, upper bound: 143.7579078
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579055, upper bound: 143.7579207
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579025, upper bound: 143.7579260
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579048, upper bound: 143.7579255
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.05
Output dim: 4, lower bound: -143.7579012, upper bound: 143.7579294

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 6.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 5.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 6.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.14 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 11.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 6.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 6.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 6.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 6.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 5.27 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 12.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.35
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0042724609375
rel_dist={4: [-143.76243730833275, 143.76243730833278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601239, upper bound: 143.7601238
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601239, upper bound: 143.7601239
time: 8.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.65
Output dim: 4, lower bound: -143.7601239, upper bound: 143.7601238
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.65
Output dim: 4, lower bound: -143.7601239, upper bound: 143.7601239

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7599292, upper bound: 143.7599283
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7599283, upper bound: 143.7599288
time: 7.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7599288, upper bound: 143.7599290
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7599283, upper bound: 143.7599292
time: 8.90 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.06
Output dim: 4, lower bound: -143.7599292, upper bound: 143.7599283
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.06
Output dim: 4, lower bound: -143.7599283, upper bound: 143.7599288
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.06
Output dim: 4, lower bound: -143.7599288, upper bound: 143.7599290
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.06
Output dim: 4, lower bound: -143.7599283, upper bound: 143.7599292

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580630, upper bound: 143.7580597
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580627, upper bound: 143.7580600
time: 7.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580600, upper bound: 143.7580627
time: 10.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580603, upper bound: 143.7580630
time: 7.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580630, upper bound: 143.7580603
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580627, upper bound: 143.7580605
time: 6.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580600, upper bound: 143.7580627
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7580597, upper bound: 143.7580630
time: 8.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580630, upper bound: 143.7580597
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580627, upper bound: 143.7580600
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580600, upper bound: 143.7580627
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580603, upper bound: 143.7580630
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580630, upper bound: 143.7580603
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580627, upper bound: 143.7580605
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580600, upper bound: 143.7580627
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.94
Output dim: 4, lower bound: -143.7580597, upper bound: 143.7580630

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578926, upper bound: 143.7578851
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578926, upper bound: 143.7578865
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578923, upper bound: 143.7578858
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578906, upper bound: 143.7578869
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578879, upper bound: 143.7578902
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578874, upper bound: 143.7578920
time: 7.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578875, upper bound: 143.7578919
time: 7.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578865, upper bound: 143.7578936
time: 8.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578919, upper bound: 143.7578865
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578919, upper bound: 143.7578875
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578902, upper bound: 143.7578874
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578902, upper bound: 143.7578879
time: 6.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578869, upper bound: 143.7578906
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578858, upper bound: 143.7578923
time: 7.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578851, upper bound: 143.7578926
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7578851, upper bound: 143.7578939
time: 7.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578926, upper bound: 143.7578851
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578926, upper bound: 143.7578865
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578923, upper bound: 143.7578858
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578906, upper bound: 143.7578869
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578879, upper bound: 143.7578902
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578874, upper bound: 143.7578920
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578875, upper bound: 143.7578919
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578865, upper bound: 143.7578936
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578919, upper bound: 143.7578865
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578919, upper bound: 143.7578875
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578902, upper bound: 143.7578874
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578902, upper bound: 143.7578879
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578869, upper bound: 143.7578906
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578858, upper bound: 143.7578923
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578851, upper bound: 143.7578926
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.44
Output dim: 4, lower bound: -143.7578851, upper bound: 143.7578939

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 6.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
time: 7.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
time: 6.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
time: 6.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
time: 6.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
time: 6.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
time: 6.14 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.67 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274033
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
time: 5.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
time: 6.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
time: 6.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
time: 6.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274033
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274045, upper bound: 143.6274067
time: 5.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274033
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274045, upper bound: 143.6274067
time: 6.02 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274088, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274055, upper bound: 143.6274009
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274075, upper bound: 143.6274026
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274036, upper bound: 143.6274033
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274033
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274045, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274033
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.11
Output dim: 4, lower bound: -143.6274045, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274050, upper bound: 143.6274067
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274023, upper bound: 143.6274080
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274080, upper bound: 143.6274023
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274067, upper bound: 143.6274050
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274033, upper bound: 143.6274075
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.11
Output dim: 4, lower bound: -143.6274009, upper bound: 143.6274088
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0042724609375
rel_dist={4: [-143.7624133928287, 143.76241339486478]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1812.81 seconds
