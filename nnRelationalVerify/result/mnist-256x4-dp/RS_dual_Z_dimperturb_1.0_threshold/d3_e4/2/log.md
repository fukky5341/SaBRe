## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 143.61867486269998


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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
execution time: IAR + RelationalAnalysis = 1.46 + 10.19 = 11.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -143.7624373, upper bound: 143.7624373

# Relational Split (RS) starts

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
time: 5.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.70
Output dim: 4, lower bound: -143.7601942, upper bound: 143.7601942
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.70
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600166
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600175, upper bound: 143.7600212
time: 6.06 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

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
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7600166, upper bound: 143.7600212
time: 5.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.99
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600166
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.99
Output dim: 4, lower bound: -143.7600175, upper bound: 143.7600212
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.99
Output dim: 4, lower bound: -143.7600212, upper bound: 143.7600175
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.99
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581578, upper bound: 143.7581483
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581483
time: 5.73 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

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

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581495, upper bound: 143.7581576
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581485, upper bound: 143.7581577
time: 5.33 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581577, upper bound: 143.7581485
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581495
time: 5.88 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

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
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581578
time: 5.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.17 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 4, lower bound: -143.7581578, upper bound: 143.7581483
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581483
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 4, lower bound: -143.7581495, upper bound: 143.7581576
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 4, lower bound: -143.7581485, upper bound: 143.7581577
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 4, lower bound: -143.7581577, upper bound: 143.7581485
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 4, lower bound: -143.7581576, upper bound: 143.7581495
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.17
Output dim: 4, lower bound: -143.7581483, upper bound: 143.7581576
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.17
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579293, upper bound: 143.7579012
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579255, upper bound: 143.7579048
time: 6.22 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

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
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579207, upper bound: 143.7579055
time: 6.89 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579078, upper bound: 143.7579188
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579071, upper bound: 143.7579250
time: 6.25 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579068, upper bound: 143.7579234
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579049, upper bound: 143.7579290
time: 7.06 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579290, upper bound: 143.7579049
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579234, upper bound: 143.7579068
time: 5.76 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

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
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579188, upper bound: 143.7579078
time: 6.32 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579055, upper bound: 143.7579207
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579025, upper bound: 143.7579260
time: 5.01 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.50 seconds

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
Output dim: 4, lower bound: -143.7579048, upper bound: 143.7579255
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.7579012, upper bound: 143.7579294
time: 6.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579293, upper bound: 143.7579012
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579255, upper bound: 143.7579048
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579260, upper bound: 143.7579025
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579207, upper bound: 143.7579055
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579078, upper bound: 143.7579188
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579071, upper bound: 143.7579250
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579068, upper bound: 143.7579234
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579049, upper bound: 143.7579290
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579290, upper bound: 143.7579049
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579234, upper bound: 143.7579068
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579250, upper bound: 143.7579071
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579188, upper bound: 143.7579078
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579055, upper bound: 143.7579207
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579025, upper bound: 143.7579260
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.01
Output dim: 4, lower bound: -143.7579048, upper bound: 143.7579255
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.01
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 4.97 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

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
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.48 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 6.66 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

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
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
time: 5.13 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

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
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 5.63 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
time: 4.65 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

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
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.82 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
time: 4.58 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 4.47 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
time: 5.24 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.48 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
time: 5.09 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.60 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
time: 5.86 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 5.38 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

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
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
time: 4.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 11.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274418
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.47
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.47
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.09 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.08 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.22 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
time: 5.24 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.42 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

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
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.43 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.44 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
time: 5.43 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.46 seconds

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
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 5.30 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 5.30 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

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
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 6.28 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
time: 6.30 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

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
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 6.27 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.41 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 6.28 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 4.94 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

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
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
time: 4.93 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=59, inp2_unstable=59, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=144, inp2_unstable=144, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=255, inp2_unstable=255, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

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
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
time: 4.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 11.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274591, upper bound: 143.6274356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274493, upper bound: 143.6274355
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274543, upper bound: 143.6274407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274437, upper bound: 143.6274418
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274480, upper bound: 143.6274428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274461, upper bound: 143.6274522
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274377, upper bound: 143.6274467
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274372, upper bound: 143.6274571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274372
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.14
Output dim: 4, lower bound: -143.6274467, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274571, upper bound: 143.6274377
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274522, upper bound: 143.6274480
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274418, upper bound: 143.6274543
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.14
Output dim: 4, lower bound: -143.6274356, upper bound: 143.6274591

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 11.65 + 597.38 = 609.03 seconds
