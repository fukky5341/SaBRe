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
execution time: IAR + RelationalAnalysis = 0.88 + 10.02 = 10.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624373, upper bound: 143.7624373

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424
time: 4.83 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.71 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.71
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.71
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830424

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830423, upper bound: 143.6830424
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830423
time: 4.43 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559862, upper bound: 143.6559862
time: 4.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559862, upper bound: 143.6559862
time: 4.69 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 18.01 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.01
Output dim: 4, lower bound: -143.6830423, upper bound: 143.6830424
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.01
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830423
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.01
Output dim: 4, lower bound: -143.6559862, upper bound: 143.6559862
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.01
Output dim: 4, lower bound: -143.6559862, upper bound: 143.6559862

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6809908, upper bound: 143.6809933
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6809909, upper bound: 143.6809924
time: 5.34 seconds

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
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830419, upper bound: 143.6830423
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830419
time: 6.04 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559860, upper bound: 143.6559818
time: 4.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559818, upper bound: 143.6559860
time: 4.50 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480036
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480036
time: 5.03 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 10.60 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6809908, upper bound: 143.6809933
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6809909, upper bound: 143.6809924
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6830419, upper bound: 143.6830423
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6830424, upper bound: 143.6830419
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6559860, upper bound: 143.6559818
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6559818, upper bound: 143.6559860
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480036
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.60
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480036

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6097712, upper bound: 143.6097712
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6097712, upper bound: 143.6097712
time: 4.12 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6809866, upper bound: 143.6809879
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6809866, upper bound: 143.6809879
time: 5.14 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559861, upper bound: 143.6559823
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559861, upper bound: 143.6559823
time: 4.75 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6558394, upper bound: 143.6558410
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6558394, upper bound: 143.6558410
time: 4.90 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559855, upper bound: 143.6559818
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559862, upper bound: 143.6559799
time: 5.16 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559818, upper bound: 143.6559825
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559815, upper bound: 143.6559860
time: 4.73 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6405464, upper bound: 143.6405440
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6405440, upper bound: 143.6405464
time: 4.67 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480033, upper bound: 143.6480036
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480033
time: 5.05 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 10.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6097712, upper bound: 143.6097712
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6097712, upper bound: 143.6097712
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6809866, upper bound: 143.6809879
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6809866, upper bound: 143.6809879
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6559861, upper bound: 143.6559823
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6559861, upper bound: 143.6559823
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6558394, upper bound: 143.6558410
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6558394, upper bound: 143.6558410
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6559855, upper bound: 143.6559818
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6559862, upper bound: 143.6559799
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6559818, upper bound: 143.6559825
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6559815, upper bound: 143.6559860
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6405464, upper bound: 143.6405440
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6405440, upper bound: 143.6405464
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6480033, upper bound: 143.6480036
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.34
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480033

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6809866, upper bound: 143.6809879
time: 5.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6809861, upper bound: 143.6809874
time: 4.98 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6706430, upper bound: 143.6706442
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6706430, upper bound: 143.6706442
time: 5.66 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559861, upper bound: 143.6559775
time: 5.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559791, upper bound: 143.6559823
time: 4.19 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480019
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480019
time: 4.95 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6350483, upper bound: 143.6350504
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6350483, upper bound: 143.6350504
time: 4.54 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382733, upper bound: 143.6382798
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6382844, upper bound: 143.6382722
time: 4.67 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559855, upper bound: 143.6559820
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559850, upper bound: 143.6559793
time: 4.96 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6406936, upper bound: 143.6406843
time: 6.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6406936, upper bound: 143.6406843
time: 5.72 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372495
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372495
time: 4.89 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559781, upper bound: 143.6559862
time: 5.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559817, upper bound: 143.6559845
time: 4.78 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6379733, upper bound: 143.6379697
time: 5.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6379757, upper bound: 143.6379694
time: 4.76 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6256302, upper bound: 143.6256307
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6256302, upper bound: 143.6256307
time: 5.07 seconds

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
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6319132, upper bound: 143.6319189
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6319194, upper bound: 143.6319150
time: 4.35 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6423627, upper bound: 143.6423582
time: 5.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6423592, upper bound: 143.6423598
time: 5.02 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.18 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6809866, upper bound: 143.6809879
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6809861, upper bound: 143.6809874
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6706430, upper bound: 143.6706442
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6706430, upper bound: 143.6706442
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6559861, upper bound: 143.6559775
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6559791, upper bound: 143.6559823
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480019
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480019
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6350483, upper bound: 143.6350504
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6350483, upper bound: 143.6350504
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6382733, upper bound: 143.6382798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6382844, upper bound: 143.6382722
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6559855, upper bound: 143.6559820
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6559850, upper bound: 143.6559793
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6406936, upper bound: 143.6406843
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6406936, upper bound: 143.6406843
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372495
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372495
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6559781, upper bound: 143.6559862
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6559817, upper bound: 143.6559845
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6379733, upper bound: 143.6379697
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6379757, upper bound: 143.6379694
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6256302, upper bound: 143.6256307
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6256302, upper bound: 143.6256307
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6319132, upper bound: 143.6319189
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6319194, upper bound: 143.6319150
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6423627, upper bound: 143.6423582
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.18
Output dim: 4, lower bound: -143.6423592, upper bound: 143.6423598

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6317734, upper bound: 143.6317732
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6317734, upper bound: 143.6317732
time: 4.22 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6546545, upper bound: 143.6546551
time: 4.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6546545, upper bound: 143.6546551
time: 4.60 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6539892, upper bound: 143.6539913
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6539892, upper bound: 143.6539913
time: 5.31 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6640556, upper bound: 143.6640562
time: 5.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6640552, upper bound: 143.6640580
time: 5.65 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6421722, upper bound: 143.6421673
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6421722, upper bound: 143.6421673
time: 4.29 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559791, upper bound: 143.6559811
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559755, upper bound: 143.6559823
time: 4.69 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480011
time: 4.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480023, upper bound: 143.6480019
time: 4.90 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480018
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6480033, upper bound: 143.6480019
time: 4.91 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6300398, upper bound: 143.6300409
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6300394, upper bound: 143.6300417
time: 5.05 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5980704, upper bound: 143.5980646
time: 5.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.5980704, upper bound: 143.5980646
time: 4.90 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6335190, upper bound: 143.6335302
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6335179, upper bound: 143.6335315
time: 5.37 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6174115, upper bound: 143.6174064
time: 5.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6174115, upper bound: 143.6174064
time: 4.07 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559843, upper bound: 143.6559818
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6559855, upper bound: 143.6559781
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6489822, upper bound: 143.6489682
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6489822, upper bound: 143.6489682
time: 5.76 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 175

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6115069, upper bound: 143.6114967
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -143.6115069, upper bound: 143.6114967
time: 4.19 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6267391, upper bound: 143.6267392
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6267391, upper bound: 143.6267392
time: 4.89 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372495
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6372346, upper bound: 143.6372494
time: 4.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372488
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6372345, upper bound: 143.6372495
time: 4.65 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 18.31 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6317734, upper bound: 143.6317732
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6317734, upper bound: 143.6317732
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6546545, upper bound: 143.6546551
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6546545, upper bound: 143.6546551
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6539892, upper bound: 143.6539913
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6539892, upper bound: 143.6539913
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6640556, upper bound: 143.6640562
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6640552, upper bound: 143.6640580
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6421722, upper bound: 143.6421673
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6421722, upper bound: 143.6421673
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6559791, upper bound: 143.6559811
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6559755, upper bound: 143.6559823
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480011
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6480023, upper bound: 143.6480019
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6480036, upper bound: 143.6480018
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6480033, upper bound: 143.6480019
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6300398, upper bound: 143.6300409
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6300394, upper bound: 143.6300417
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.5980704, upper bound: 143.5980646
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.5980704, upper bound: 143.5980646
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6335190, upper bound: 143.6335302
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6335179, upper bound: 143.6335315
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6174115, upper bound: 143.6174064
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6174115, upper bound: 143.6174064
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6559843, upper bound: 143.6559818
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6559855, upper bound: 143.6559781
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6489822, upper bound: 143.6489682
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6489822, upper bound: 143.6489682
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6115069, upper bound: 143.6114967
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6115069, upper bound: 143.6114967
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6267391, upper bound: 143.6267392
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6267391, upper bound: 143.6267392
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372495
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6372346, upper bound: 143.6372494
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6372348, upper bound: 143.6372488
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 18.31
Output dim: 4, lower bound: -143.6372345, upper bound: 143.6372495
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6559781, upper bound: 143.6559862
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6559817, upper bound: 143.6559845
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6379733, upper bound: 143.6379697
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6379757, upper bound: 143.6379694
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6256302, upper bound: 143.6256307
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6256302, upper bound: 143.6256307
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6319132, upper bound: 143.6319189
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6319194, upper bound: 143.6319150
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6423627, upper bound: 143.6423582
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.31
Output dim: 4, lower bound: -143.6423592, upper bound: 143.6423598

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 10.90 + 593.94 = 604.84 seconds
