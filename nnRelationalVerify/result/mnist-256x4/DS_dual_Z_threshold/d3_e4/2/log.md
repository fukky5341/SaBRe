## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 143.61867486269998


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

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.88 + 10.05 = 10.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624373, upper bound: 143.7624373

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 5.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 5.55 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.50 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.50
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.50
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600166
time: 6.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600175, upper bound: 143.7600212
time: 5.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600175
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600166, upper bound: 143.7600212
time: 5.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 12.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600166
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 4, lower bound: -143.7600175, upper bound: 143.7600212
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600175
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 4, lower bound: -143.7600166, upper bound: 143.7600212

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581578, upper bound: 143.7581483
time: 5.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581483
time: 5.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581495, upper bound: 143.7581576
time: 5.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581485, upper bound: 143.7581577
time: 5.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581577, upper bound: 143.7581485
time: 5.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581495
time: 5.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581576
time: 5.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581578
time: 5.32 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 12.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581578, upper bound: 143.7581483
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581483
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581495, upper bound: 143.7581576
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581485, upper bound: 143.7581577
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581577, upper bound: 143.7581485
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581495
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581576
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.07
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581578

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579293, upper bound: 143.7579012
time: 6.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579255, upper bound: 143.7579048
time: 6.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579260, upper bound: 143.7579025
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579207, upper bound: 143.7579055
time: 6.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579078, upper bound: 143.7579188
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579071, upper bound: 143.7579250
time: 6.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579068, upper bound: 143.7579234
time: 6.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579049, upper bound: 143.7579290
time: 6.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579290, upper bound: 143.7579049
time: 6.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579234, upper bound: 143.7579068
time: 5.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579250, upper bound: 143.7579071
time: 5.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579188, upper bound: 143.7579078
time: 6.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579055, upper bound: 143.7579207
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579025, upper bound: 143.7579260
time: 4.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579048, upper bound: 143.7579255
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579012, upper bound: 143.7579294
time: 6.63 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 12.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579293, upper bound: 143.7579012
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579255, upper bound: 143.7579048
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579260, upper bound: 143.7579025
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579207, upper bound: 143.7579055
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579078, upper bound: 143.7579188
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579071, upper bound: 143.7579250
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579068, upper bound: 143.7579234
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579049, upper bound: 143.7579290
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579290, upper bound: 143.7579049
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579234, upper bound: 143.7579068
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579250, upper bound: 143.7579071
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579188, upper bound: 143.7579078
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579055, upper bound: 143.7579207
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579025, upper bound: 143.7579260
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579048, upper bound: 143.7579255
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.77
Output dim: 4, lower bound: -143.7579012, upper bound: 143.7579294

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 4.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 6.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 4.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 5.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 4.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 4.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 4.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 4.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 4.76 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 10.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.35
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 4.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 4.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 4.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 5.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 5.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 6.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 6.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 6.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 6.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 5.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 4.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
time: 4.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
time: 4.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
time: 4.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
time: 4.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
time: 4.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -76.0228043, 60.6544914, -76.0228043, 60.6544914, -136.6772766, 136.6772766
1: -62.9943390, 53.7792549, -62.9943390, 53.7792549, -116.7735901, 116.7735901
2: -83.4031906, 54.4029045, -83.4031906, 54.4029045, -137.8060913, 137.8060913
3: -88.7539597, 46.9208794, -88.7539597, 46.9208794, -135.6748352, 135.6748352
4: -81.9910431, 63.0132256, -81.9910431, 63.0132256, -145.0042725, 145.0042725
5: -73.1580811, 56.9551659, -73.1580811, 56.9551659, -130.1132507, 130.1132507
6: -70.2768860, 67.0410385, -70.2768860, 67.0410385, -137.3179321, 137.3179321
7: -76.0090866, 64.0053253, -76.0090866, 64.0053253, -140.0144043, 140.0144043
8: -92.0233536, 63.7623940, -92.0233536, 63.7623940, -155.7857513, 155.7857513
9: -69.3594666, 68.1468353, -69.3594666, 68.1468353, -137.5062561, 137.5062561

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274461
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274428, upper bound: 143.6274480
time: 4.50 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 10.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274461
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.22
Output dim: 4, lower bound: -143.6274428, upper bound: 143.6274480
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.22
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 10.93 + 593.09 = 604.02 seconds
