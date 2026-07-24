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
execution time: IAR + LP analysis = 1.30 + 8.30 = 9.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624614, upper bound: 143.7624614


# Binary Search by BASE starts (time budget: 1990.41 seconds, max iter: 100)

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
Binary search time: 33.33 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1957.08 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7546576, upper bound: 143.7546576
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7546576, upper bound: 143.7546577
time: 5.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.13
Output dim: 4, lower bound: -143.7546576, upper bound: 143.7546576
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.13
Output dim: 4, lower bound: -143.7546576, upper bound: 143.7546577

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7546567, upper bound: 143.7546577
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7546577, upper bound: 143.7546567
time: 5.65 seconds

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
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7359743, upper bound: 143.7359743
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7359743, upper bound: 143.7359743
time: 5.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.80
Output dim: 4, lower bound: -143.7546567, upper bound: 143.7546577
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.80
Output dim: 4, lower bound: -143.7546577, upper bound: 143.7546567
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.80
Output dim: 4, lower bound: -143.7359743, upper bound: 143.7359743
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.80
Output dim: 4, lower bound: -143.7359743, upper bound: 143.7359743

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7275010, upper bound: 143.7275028
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7275010, upper bound: 143.7275028
time: 5.97 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6666944, upper bound: 143.6666852
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6666944, upper bound: 143.6666852
time: 4.78 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7037088, upper bound: 143.7037089
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7037088, upper bound: 143.7037089
time: 4.84 seconds

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
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7359743, upper bound: 143.7359725
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7359725, upper bound: 143.7359743
time: 6.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.7275010, upper bound: 143.7275028
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.7275010, upper bound: 143.7275028
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.6666944, upper bound: 143.6666852
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.6666944, upper bound: 143.6666852
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.7037088, upper bound: 143.7037089
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.7037088, upper bound: 143.7037089
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.7359743, upper bound: 143.7359725
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 4, lower bound: -143.7359725, upper bound: 143.7359743

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7155063, upper bound: 143.7155070
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7155063, upper bound: 143.7155070
time: 5.68 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6907450, upper bound: 143.6907427
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6907450, upper bound: 143.6907427
time: 5.04 seconds

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
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6065814, upper bound: 143.6065845
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6065814, upper bound: 143.6065845
time: 4.49 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6415621, upper bound: 143.6415590
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6415621, upper bound: 143.6415590
time: 5.98 seconds

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
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6835003, upper bound: 143.6835003
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6835003, upper bound: 143.6835003
time: 6.07 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6790627, upper bound: 143.6790627
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6790627, upper bound: 143.6790627
time: 5.21 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7255212, upper bound: 143.7255119
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7255212, upper bound: 143.7255119
time: 4.80 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7305527, upper bound: 143.7305464
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7305430, upper bound: 143.7305579
time: 5.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.7155063, upper bound: 143.7155070
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.7155063, upper bound: 143.7155070
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6907450, upper bound: 143.6907427
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6907450, upper bound: 143.6907427
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6065814, upper bound: 143.6065845
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6065814, upper bound: 143.6065845
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6415621, upper bound: 143.6415590
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6415621, upper bound: 143.6415590
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6835003, upper bound: 143.6835003
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6835003, upper bound: 143.6835003
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6790627, upper bound: 143.6790627
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.6790627, upper bound: 143.6790627
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.7255212, upper bound: 143.7255119
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.7255212, upper bound: 143.7255119
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.7305527, upper bound: 143.7305464
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 4, lower bound: -143.7305430, upper bound: 143.7305579

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6399517, upper bound: 143.6399522
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6399517, upper bound: 143.6399522
time: 5.23 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7154321, upper bound: 143.7154297
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7154298, upper bound: 143.7154319
time: 5.82 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6845542, upper bound: 143.6845489
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6845490, upper bound: 143.6845534
time: 5.15 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6907187, upper bound: 143.6907099
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6907103, upper bound: 143.6907163
time: 4.49 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5928127, upper bound: 143.5928146
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5928127, upper bound: 143.5928146
time: 5.46 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6411706, upper bound: 143.6411650
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6411705, upper bound: 143.6411685
time: 4.64 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6834945, upper bound: 143.6835003
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6835003, upper bound: 143.6834945
time: 5.00 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6778521, upper bound: 143.6778492
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6778492, upper bound: 143.6778521
time: 5.01 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6790627, upper bound: 143.6790608
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6790608, upper bound: 143.6790627
time: 5.35 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282902
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282902
time: 4.40 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7253882, upper bound: 143.7253372
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7253496, upper bound: 143.7253819
time: 5.87 seconds

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
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7021006, upper bound: 143.7020982
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7021006, upper bound: 143.7020982
time: 5.05 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7302797, upper bound: 143.7302749
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7302783, upper bound: 143.7302753
time: 5.14 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6978796, upper bound: 143.6978817
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6978796, upper bound: 143.6978817
time: 5.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 11.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6399517, upper bound: 143.6399522
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6399517, upper bound: 143.6399522
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7154321, upper bound: 143.7154297
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7154298, upper bound: 143.7154319
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6845542, upper bound: 143.6845489
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6845490, upper bound: 143.6845534
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6907187, upper bound: 143.6907099
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6907103, upper bound: 143.6907163
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.5928127, upper bound: 143.5928146
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.5928127, upper bound: 143.5928146
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6411706, upper bound: 143.6411650
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6411705, upper bound: 143.6411685
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6834945, upper bound: 143.6835003
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6835003, upper bound: 143.6834945
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6778521, upper bound: 143.6778492
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6778492, upper bound: 143.6778521
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6790627, upper bound: 143.6790608
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6790608, upper bound: 143.6790627
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282902
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282902
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7253882, upper bound: 143.7253372
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7253496, upper bound: 143.7253819
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7021006, upper bound: 143.7020982
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7021006, upper bound: 143.7020982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7302797, upper bound: 143.7302749
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.7302783, upper bound: 143.7302753
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6978796, upper bound: 143.6978817
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.34
Output dim: 4, lower bound: -143.6978796, upper bound: 143.6978817

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6399517, upper bound: 143.6399488
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6399486, upper bound: 143.6399522
time: 4.32 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6365892, upper bound: 143.6365990
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6365912, upper bound: 143.6365949
time: 5.64 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7154321, upper bound: 143.7154290
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7154308, upper bound: 143.7154297
time: 6.37 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7097501, upper bound: 143.7097731
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7097501, upper bound: 143.7097731
time: 5.64 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6825797, upper bound: 143.6825870
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6825848, upper bound: 143.6825847
time: 5.53 seconds

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
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6823382, upper bound: 143.6823506
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6823382, upper bound: 143.6823509
time: 6.20 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6803442, upper bound: 143.6803092
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6803442, upper bound: 143.6803092
time: 5.87 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6123263, upper bound: 143.6123347
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6123263, upper bound: 143.6123347
time: 4.92 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6393117, upper bound: 143.6393025
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6393117, upper bound: 143.6393025
time: 6.21 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6411668, upper bound: 143.6411685
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6411705, upper bound: 143.6411658
time: 5.51 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6766180, upper bound: 143.6766307
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6766317, upper bound: 143.6766183
time: 5.42 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6799815, upper bound: 143.6799847
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6799896, upper bound: 143.6799787
time: 5.78 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6486119, upper bound: 143.6486083
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6486119, upper bound: 143.6486082
time: 6.05 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6228297, upper bound: 143.6228405
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6228297, upper bound: 143.6228405
time: 6.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6484358, upper bound: 143.6484433
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6484358, upper bound: 143.6484433
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282882
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282882
time: 6.04 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6399517, upper bound: 143.6399488
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6399486, upper bound: 143.6399522
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6365892, upper bound: 143.6365990
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6365912, upper bound: 143.6365949
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.7154321, upper bound: 143.7154290
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.7154308, upper bound: 143.7154297
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.7097501, upper bound: 143.7097731
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.7097501, upper bound: 143.7097731
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6825797, upper bound: 143.6825870
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6825848, upper bound: 143.6825847
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6823382, upper bound: 143.6823506
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6823382, upper bound: 143.6823509
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6803442, upper bound: 143.6803092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6803442, upper bound: 143.6803092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6123263, upper bound: 143.6123347
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6123263, upper bound: 143.6123347
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6393117, upper bound: 143.6393025
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6393117, upper bound: 143.6393025
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6411668, upper bound: 143.6411685
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6411705, upper bound: 143.6411658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6766180, upper bound: 143.6766307
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6766317, upper bound: 143.6766183
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6799815, upper bound: 143.6799847
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6799896, upper bound: 143.6799787
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6486119, upper bound: 143.6486083
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6486119, upper bound: 143.6486082
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6228297, upper bound: 143.6228405
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6228297, upper bound: 143.6228405
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6484358, upper bound: 143.6484433
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6484358, upper bound: 143.6484433
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282882
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.45
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282882
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282902
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.6283011, upper bound: 143.6282902
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.7253882, upper bound: 143.7253372
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.7253496, upper bound: 143.7253819
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.7021006, upper bound: 143.7020982
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.7021006, upper bound: 143.7020982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.7302797, upper bound: 143.7302749
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.7302783, upper bound: 143.7302753
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.6978796, upper bound: 143.6978817
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.45
Output dim: 4, lower bound: -143.6978796, upper bound: 143.6978817
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=145.0042724609375
rel_dist={4: [-143.76245312009848, 143.76245312009854]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424
time: 5.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.42
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.42
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830421
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830421, upper bound: 143.6830424
time: 6.42 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658300
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658300
time: 4.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.39
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830421
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.39
Output dim: 4, lower bound: -143.6830421, upper bound: 143.6830424
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.39
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658300
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.39
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658300

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6035938, upper bound: 143.6035941
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6035938, upper bound: 143.6035941
time: 5.12 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300
time: 5.75 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658298, upper bound: 143.6658300
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658298
time: 5.66 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658284
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300
time: 5.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6035938, upper bound: 143.6035941
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6035938, upper bound: 143.6035941
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6658298, upper bound: 143.6658300
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658298
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658284
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.92
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5776008, upper bound: 143.5776051
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5776008, upper bound: 143.5776051
time: 5.20 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658283, upper bound: 143.6658293
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300
time: 6.04 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610215, upper bound: 143.6610211
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610211, upper bound: 143.6610213
time: 6.05 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658294, upper bound: 143.6658298
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658295
time: 5.33 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658298, upper bound: 143.6658284
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658280
time: 5.44 seconds

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658280, upper bound: 143.6658300
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658298
time: 5.09 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.31 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.5776008, upper bound: 143.5776051
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.5776008, upper bound: 143.5776051
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658283, upper bound: 143.6658293
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658300
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6610215, upper bound: 143.6610211
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6610211, upper bound: 143.6610213
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658294, upper bound: 143.6658298
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658295
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658298, upper bound: 143.6658284
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658300, upper bound: 143.6658280
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658280, upper bound: 143.6658300
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.31
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658298

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6483020, upper bound: 143.6482976
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6483020, upper bound: 143.6482976
time: 4.72 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658276, upper bound: 143.6658300
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658289
time: 5.47 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6568226, upper bound: 143.6568238
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6568230, upper bound: 143.6568238
time: 5.42 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610211, upper bound: 143.6610211
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610210, upper bound: 143.6610213
time: 4.29 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6590197, upper bound: 143.6590162
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6590157, upper bound: 143.6590203
time: 5.13 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6153090, upper bound: 143.6153080
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6153090, upper bound: 143.6153080
time: 4.70 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6153106, upper bound: 143.6153054
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6153106, upper bound: 143.6153054
time: 5.51 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382835, upper bound: 143.6382702
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382835, upper bound: 143.6382702
time: 6.75 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658280, upper bound: 143.6658298
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658277, upper bound: 143.6658300
time: 5.04 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658296
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658276, upper bound: 143.6658298
time: 4.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 10.99 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6483020, upper bound: 143.6482976
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6483020, upper bound: 143.6482976
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6658276, upper bound: 143.6658300
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658289
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6568226, upper bound: 143.6568238
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6568230, upper bound: 143.6568238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6610211, upper bound: 143.6610211
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6610210, upper bound: 143.6610213
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6590197, upper bound: 143.6590162
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6590157, upper bound: 143.6590203
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6153090, upper bound: 143.6153080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6153090, upper bound: 143.6153080
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6153106, upper bound: 143.6153054
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6153106, upper bound: 143.6153054
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6382835, upper bound: 143.6382702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6382835, upper bound: 143.6382702
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6658280, upper bound: 143.6658298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6658277, upper bound: 143.6658300
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658296
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 10.99
Output dim: 4, lower bound: -143.6658276, upper bound: 143.6658298

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6217327, upper bound: 143.6217327
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6217327, upper bound: 143.6217327
time: 5.89 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6443428, upper bound: 143.6443383
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6443428, upper bound: 143.6443384
time: 4.75 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610212, upper bound: 143.6610208
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610210, upper bound: 143.6610210
time: 5.10 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382705, upper bound: 143.6382712
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382705, upper bound: 143.6382712
time: 4.89 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6487432, upper bound: 143.6487412
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6487432, upper bound: 143.6487412
time: 5.22 seconds

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
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6568223, upper bound: 143.6568238
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6568230, upper bound: 143.6568238
time: 4.82 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6541192, upper bound: 143.6541124
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6541087, upper bound: 143.6541178
time: 5.47 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6394276, upper bound: 143.6394273
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6394276, upper bound: 143.6394273
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6080070, upper bound: 143.6080062
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6080070, upper bound: 143.6080062
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6192502, upper bound: 143.6192545
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6192502, upper bound: 143.6192545
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382835, upper bound: 143.6382691
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382810, upper bound: 143.6382702
time: 4.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 107

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6295798, upper bound: 143.6295794
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6295798, upper bound: 143.6295794
time: 5.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6270456, upper bound: 143.6270431
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6270456, upper bound: 143.6270431
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6585955, upper bound: 143.6585983
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6585955, upper bound: 143.6585983
time: 6.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658283, upper bound: 143.6658282
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658296
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6522927, upper bound: 143.6522920
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6522927, upper bound: 143.6522920
time: 4.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 10.40 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6217327, upper bound: 143.6217327
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6217327, upper bound: 143.6217327
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6443428, upper bound: 143.6443383
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6443428, upper bound: 143.6443384
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6610212, upper bound: 143.6610208
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6610210, upper bound: 143.6610210
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6382705, upper bound: 143.6382712
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6382705, upper bound: 143.6382712
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6487432, upper bound: 143.6487412
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6487432, upper bound: 143.6487412
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6568223, upper bound: 143.6568238
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6568230, upper bound: 143.6568238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6541192, upper bound: 143.6541124
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6541087, upper bound: 143.6541178
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6394276, upper bound: 143.6394273
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6394276, upper bound: 143.6394273
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6080070, upper bound: 143.6080062
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6080070, upper bound: 143.6080062
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6192502, upper bound: 143.6192545
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6192502, upper bound: 143.6192545
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6382835, upper bound: 143.6382691
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6382810, upper bound: 143.6382702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6295798, upper bound: 143.6295794
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6295798, upper bound: 143.6295794
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6270456, upper bound: 143.6270431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6270456, upper bound: 143.6270431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6585955, upper bound: 143.6585983
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6585955, upper bound: 143.6585983
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6658283, upper bound: 143.6658282
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6658284, upper bound: 143.6658296
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6522927, upper bound: 143.6522920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.40
Output dim: 4, lower bound: -143.6522927, upper bound: 143.6522920

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5590750, upper bound: 143.5590750
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5590750, upper bound: 143.5590750
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6175981, upper bound: 143.6176004
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6175970, upper bound: 143.6176004
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6256020, upper bound: 143.6256068
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6256020, upper bound: 143.6256068
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6308136, upper bound: 143.6308122
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6308136, upper bound: 143.6308122
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610212, upper bound: 143.6610210
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6610210, upper bound: 143.6610208
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=145.0042724609375
rel_dist={4: [-143.76243730833275, 143.76243730833278]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7624134, upper bound: 143.7624134
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7624134, upper bound: 143.7624134
time: 8.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 16.05 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 16.05
Output dim: 4, lower bound: -143.7624134, upper bound: 143.7624134
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 16.05
Output dim: 4, lower bound: -143.7624134, upper bound: 143.7624134

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
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7216055, upper bound: 143.7216057
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7216055, upper bound: 143.7216057
time: 6.26 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7487794, upper bound: 143.7487785
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7487794, upper bound: 143.7487785
time: 6.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.63
Output dim: 4, lower bound: -143.7216055, upper bound: 143.7216057
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.63
Output dim: 4, lower bound: -143.7216055, upper bound: 143.7216057
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.63
Output dim: 4, lower bound: -143.7487794, upper bound: 143.7487785
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.63
Output dim: 4, lower bound: -143.7487794, upper bound: 143.7487785

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6754613, upper bound: 143.6754614
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6754613, upper bound: 143.6754614
time: 6.94 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7216051, upper bound: 143.7216058
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7216055, upper bound: 143.7216054
time: 6.51 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323050, upper bound: 143.7323044
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323050, upper bound: 143.7323044
time: 6.70 seconds

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7371269, upper bound: 143.7371247
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7371269, upper bound: 143.7371247
time: 7.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.6754613, upper bound: 143.6754614
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.6754613, upper bound: 143.6754614
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.7216051, upper bound: 143.7216058
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.7216055, upper bound: 143.7216054
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.7323050, upper bound: 143.7323044
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.7323050, upper bound: 143.7323044
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.7371269, upper bound: 143.7371247
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.42
Output dim: 4, lower bound: -143.7371269, upper bound: 143.7371247

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6754611, upper bound: 143.6754614
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6754613, upper bound: 143.6754614
time: 5.28 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
time: 5.89 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7216051, upper bound: 143.7216052
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7216044, upper bound: 143.7216057
time: 6.74 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6987702, upper bound: 143.6987701
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6987702, upper bound: 143.6987701
time: 6.48 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323046, upper bound: 143.7323044
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323049, upper bound: 143.7323040
time: 6.46 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6791577, upper bound: 143.6791577
time: 8.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6791577, upper bound: 143.6791577
time: 8.92 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5866155, upper bound: 143.5866156
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5866155, upper bound: 143.5866156
time: 5.67 seconds

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
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323260, upper bound: 143.7323206
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323223
time: 7.85 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 17.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6754611, upper bound: 143.6754614
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6754613, upper bound: 143.6754614
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.7216051, upper bound: 143.7216052
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.7216044, upper bound: 143.7216057
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6987702, upper bound: 143.6987701
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6987702, upper bound: 143.6987701
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.7323046, upper bound: 143.7323044
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.7323049, upper bound: 143.7323040
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6791577, upper bound: 143.6791577
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.6791577, upper bound: 143.6791577
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.5866155, upper bound: 143.5866156
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.5866155, upper bound: 143.5866156
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.7323260, upper bound: 143.7323206
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 17.00
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323223

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6611435, upper bound: 143.6611429
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6611435, upper bound: 143.6611426
time: 5.55 seconds

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6753505, upper bound: 143.6753501
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6753502, upper bound: 143.6753505
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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
time: 6.12 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6278945, upper bound: 143.6278945
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6278945, upper bound: 143.6278945
time: 5.21 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7116576, upper bound: 143.7116569
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7116576, upper bound: 143.7116569
time: 7.15 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 159

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6843890, upper bound: 143.6843889
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6843890, upper bound: 143.6843889
time: 6.41 seconds

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6987702, upper bound: 143.6987687
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6987691, upper bound: 143.6987701
time: 5.78 seconds

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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6923645, upper bound: 143.6923644
time: 7.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6923646, upper bound: 143.6923641
time: 6.85 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6777980, upper bound: 143.6777959
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6777980, upper bound: 143.6777959
time: 6.65 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6922895, upper bound: 143.6922897
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6922895, upper bound: 143.6922897
time: 4.95 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6716751, upper bound: 143.6716739
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6716739, upper bound: 143.6716751
time: 6.77 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 124

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6646569, upper bound: 143.6646568
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6646569, upper bound: 143.6646568
time: 6.21 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7322093, upper bound: 143.7322059
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7322094, upper bound: 143.7322054
time: 7.95 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323219
time: 11.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323223
time: 7.81 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6611435, upper bound: 143.6611429
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6611435, upper bound: 143.6611426
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6753505, upper bound: 143.6753501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6753502, upper bound: 143.6753505
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6333550, upper bound: 143.6333550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6278945, upper bound: 143.6278945
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6278945, upper bound: 143.6278945
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.7116576, upper bound: 143.7116569
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.7116576, upper bound: 143.7116569
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6843890, upper bound: 143.6843889
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6843890, upper bound: 143.6843889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6987702, upper bound: 143.6987687
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6987691, upper bound: 143.6987701
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6923645, upper bound: 143.6923644
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6923646, upper bound: 143.6923641
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6777980, upper bound: 143.6777959
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6777980, upper bound: 143.6777959
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6922895, upper bound: 143.6922897
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6922895, upper bound: 143.6922897
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6716751, upper bound: 143.6716739
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6716739, upper bound: 143.6716751
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6646569, upper bound: 143.6646568
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.6646569, upper bound: 143.6646568
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.7322093, upper bound: 143.7322059
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.7322094, upper bound: 143.7322054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323219
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.30
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323223

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176230
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176230
time: 5.75 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6605015, upper bound: 143.6605001
time: 6.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6605008, upper bound: 143.6605001
time: 5.77 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6672024, upper bound: 143.6672041
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6672024, upper bound: 143.6672041
time: 6.70 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6753502, upper bound: 143.6753505
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6753502, upper bound: 143.6753504
time: 6.22 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176230
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176227
time: 6.40 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6333253, upper bound: 143.6333253
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6333253, upper bound: 143.6333253
time: 5.90 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6247928, upper bound: 143.6247928
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6247928, upper bound: 143.6247928
time: 6.91 seconds

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

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6228169, upper bound: 143.6228169
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6228169, upper bound: 143.6228169
time: 5.42 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 124
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7112141, upper bound: 143.7112139
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7112142, upper bound: 143.7112139
time: 6.34 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176230
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176230
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6605015, upper bound: 143.6605001
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6605008, upper bound: 143.6605001
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6672024, upper bound: 143.6672041
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6672024, upper bound: 143.6672041
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6753502, upper bound: 143.6753505
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6753502, upper bound: 143.6753504
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176230
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6176227, upper bound: 143.6176227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6333253, upper bound: 143.6333253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6333253, upper bound: 143.6333253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6247928, upper bound: 143.6247928
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6247928, upper bound: 143.6247928
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6228169, upper bound: 143.6228169
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.6228169, upper bound: 143.6228169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.7112141, upper bound: 143.7112139
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.31
Output dim: 4, lower bound: -143.7112142, upper bound: 143.7112139
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.7116576, upper bound: 143.7116569
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6843890, upper bound: 143.6843889
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6843890, upper bound: 143.6843889
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6987702, upper bound: 143.6987687
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6987691, upper bound: 143.6987701
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6923645, upper bound: 143.6923644
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6923646, upper bound: 143.6923641
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6777980, upper bound: 143.6777959
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6777980, upper bound: 143.6777959
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6922895, upper bound: 143.6922897
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6922895, upper bound: 143.6922897
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6716751, upper bound: 143.6716739
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6716739, upper bound: 143.6716751
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6646569, upper bound: 143.6646568
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.6646569, upper bound: 143.6646568
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.7322093, upper bound: 143.7322059
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.7322094, upper bound: 143.7322054
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323219
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.31
Output dim: 4, lower bound: -143.7323254, upper bound: 143.7323223
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=145.0042724609375
rel_dist={4: [-143.7624133928287, 143.76241339486478]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1807.86 seconds
