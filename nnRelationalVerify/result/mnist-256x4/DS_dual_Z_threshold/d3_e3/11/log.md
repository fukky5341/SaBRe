## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 73.7928750582


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820)
1: (-34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384)
2: (-47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675)
3: (-53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606)
4: (-47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160)
5: (-42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925)
6: (-40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805)
7: (-44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109)
8: (-54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457)
9: (-42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.30 + 10.54 = 12.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626623
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626623
time: 7.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.02 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.02
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626623
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.02
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626623

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8626607, upper bound: 73.8626623
time: 7.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626607
time: 10.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8626607, upper bound: 73.8626623
time: 6.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626607
time: 8.26 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 17.01 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.01
Output dim: 9, lower bound: -73.8626607, upper bound: 73.8626623
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.01
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626607
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.01
Output dim: 9, lower bound: -73.8626607, upper bound: 73.8626623
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.01
Output dim: 9, lower bound: -73.8626623, upper bound: 73.8626607

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589031
time: 7.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589026
time: 7.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588958
time: 7.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588959
time: 6.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
time: 6.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589026
time: 7.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588958
time: 6.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588959
time: 6.39 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589031
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589026
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588958
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588959
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589026
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.81
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588959

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
time: 7.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
time: 7.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589024
time: 6.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589026
time: 8.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588959
time: 8.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588959
time: 5.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588960
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588958
time: 5.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
time: 7.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589032
time: 7.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589024
time: 7.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589028
time: 6.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588958
time: 7.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588959
time: 6.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588960
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588958
time: 6.13 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589024
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589026
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588959
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588959
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588960
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588958
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589031
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8589032
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589024
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8589028
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588959
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588960
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.51
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588958

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588939, upper bound: 73.8589030
time: 7.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588961
time: 7.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588938, upper bound: 73.8589031
time: 7.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
time: 7.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589025
time: 5.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588961
time: 6.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589026
time: 6.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8588963
time: 8.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
time: 7.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588930
time: 6.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588959
time: 7.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588930
time: 7.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
time: 7.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588938
time: 7.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
time: 7.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588938
time: 6.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588939, upper bound: 73.8589031
time: 7.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
time: 6.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588938, upper bound: 73.8589031
time: 6.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
time: 6.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589025
time: 7.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588963
time: 6.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589026
time: 7.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588963
time: 7.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
time: 7.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588931
time: 8.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
time: 7.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588930
time: 7.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
time: 7.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588937
time: 8.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
time: 6.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588938
time: 6.72 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588939, upper bound: 73.8589030
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588961
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588938, upper bound: 73.8589031
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589025
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588961
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589026
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8588963
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588959
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588938
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588939, upper bound: 73.8589031
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588938, upper bound: 73.8589031
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589025
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588963
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589026
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588963
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588931
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588937
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.93
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588938

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570276
time: 8.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570276
time: 7.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570265
time: 9.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570265
time: 8.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570278
time: 6.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570278
time: 7.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570267
time: 6.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570267
time: 7.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8349380, 36.4853439, -43.8349380, 36.4853439, -80.3202820, 80.3202820
1: -34.8081284, 31.8318081, -34.8081284, 31.8318081, -66.6399384, 66.6399384
2: -47.5706978, 32.2634697, -47.5706978, 32.2634697, -79.8341675, 79.8341675
3: -53.0499039, 26.8578606, -53.0499039, 26.8578606, -79.9077606, 79.9077606
4: -47.9338379, 35.7787933, -47.9338379, 35.7787933, -83.7126160, 83.7126160
5: -42.2251434, 33.0509529, -42.2251434, 33.0509529, -75.2760925, 75.2760925
6: -40.8981514, 39.3887291, -40.8981514, 39.3887291, -80.2868805, 80.2868805
7: -44.6253395, 37.3742828, -44.6253395, 37.3742828, -81.9996109, 81.9996109
8: -54.3321075, 36.0625381, -54.3321075, 36.0625381, -90.3946457, 90.3946457
9: -42.8744965, 37.5781250, -42.8744965, 37.5781250, -80.4526062, 80.4526062

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570255, upper bound: 73.8570275
time: 7.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8570255, upper bound: 73.8570275
time: 7.05 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 16.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570276
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570276
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570265
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570265
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570254, upper bound: 73.8570278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570267
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570260, upper bound: 73.8570267
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570255, upper bound: 73.8570275
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.69
Output dim: 9, lower bound: -73.8570255, upper bound: 73.8570275
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588961
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589026
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8588963
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588930
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588959
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588938
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588938
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588939, upper bound: 73.8589031
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588938, upper bound: 73.8589031
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588960, upper bound: 73.8588963
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589025
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588963
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588931, upper bound: 73.8589026
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588959, upper bound: 73.8588963
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8589028, upper bound: 73.8588931
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8589025, upper bound: 73.8588930
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588960
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8589032, upper bound: 73.8588937
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8588963, upper bound: 73.8588958
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 9, lower bound: -73.8589031, upper bound: 73.8588938

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 12.84 + 599.68 = 612.51 seconds
