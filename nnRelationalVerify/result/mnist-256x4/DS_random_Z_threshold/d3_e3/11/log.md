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
execution time: IAR + RelationalAnalysis = 0.90 + 10.46 = 11.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -73.8667418, upper bound: 73.8667418

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477940
time: 6.82 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477940
time: 6.72 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.56 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.56
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477940
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.56
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477940

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477939
time: 7.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477940
time: 7.05 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8450285, upper bound: 73.8450291
time: 7.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8450291, upper bound: 73.8450285
time: 7.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.80 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.80
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477939
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.80
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477940
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.80
Output dim: 9, lower bound: -73.8450285, upper bound: 73.8450291
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.80
Output dim: 9, lower bound: -73.8450291, upper bound: 73.8450285

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477931, upper bound: 73.8477938
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477929
time: 6.66 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477940
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477940
time: 6.32 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8396659, upper bound: 73.8396694
time: 6.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8396670, upper bound: 73.8396687
time: 6.85 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8450291, upper bound: 73.8450278
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8450285, upper bound: 73.8450285
time: 6.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.24 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8477931, upper bound: 73.8477938
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8477940, upper bound: 73.8477929
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477940
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477940
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8396659, upper bound: 73.8396694
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8396670, upper bound: 73.8396687
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8450291, upper bound: 73.8450278
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.24
Output dim: 9, lower bound: -73.8450285, upper bound: 73.8450285

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8469843, upper bound: 73.8469855
time: 7.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8469841, upper bound: 73.8469857
time: 7.27 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8390267, upper bound: 73.8390267
time: 6.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8390267, upper bound: 73.8390267
time: 7.32 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 216

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8418338, upper bound: 73.8418338
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8418338, upper bound: 73.8418338
time: 6.35 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477915, upper bound: 73.8477940
time: 5.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477908
time: 5.02 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8395293, upper bound: 73.8395370
time: 7.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8395265, upper bound: 73.8395379
time: 7.28 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8396615, upper bound: 73.8396687
time: 6.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8396670, upper bound: 73.8396616
time: 7.75 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 64

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8439222, upper bound: 73.8439208
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8439222, upper bound: 73.8439208
time: 6.62 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8450265, upper bound: 73.8450285
time: 6.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8450286, upper bound: 73.8450274
time: 7.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 18.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8469843, upper bound: 73.8469855
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8469841, upper bound: 73.8469857
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8390267, upper bound: 73.8390267
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8390267, upper bound: 73.8390267
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8418338, upper bound: 73.8418338
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8418338, upper bound: 73.8418338
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8477915, upper bound: 73.8477940
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477908
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8395293, upper bound: 73.8395370
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8395265, upper bound: 73.8395379
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8396615, upper bound: 73.8396687
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8396670, upper bound: 73.8396616
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8439222, upper bound: 73.8439208
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8439222, upper bound: 73.8439208
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8450265, upper bound: 73.8450285
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.62
Output dim: 9, lower bound: -73.8450286, upper bound: 73.8450274

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8469834, upper bound: 73.8469855
time: 6.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8469843, upper bound: 73.8469844
time: 6.72 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8468225, upper bound: 73.8468248
time: 6.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8468225, upper bound: 73.8468248
time: 6.33 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8385178, upper bound: 73.8385164
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8385179, upper bound: 73.8385162
time: 7.44 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8330429, upper bound: 73.8330442
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8330430, upper bound: 73.8330442
time: 5.81 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8418305, upper bound: 73.8418338
time: 7.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8418338, upper bound: 73.8418301
time: 6.57 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8371311, upper bound: 73.8371295
time: 7.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8371311, upper bound: 73.8371295
time: 7.47 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477915, upper bound: 73.8477930
time: 6.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477914, upper bound: 73.8477940
time: 6.96 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477901
time: 6.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8477937, upper bound: 73.8477908
time: 6.98 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8395267, upper bound: 73.8395370
time: 5.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8395293, upper bound: 73.8395339
time: 6.01 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8395265, upper bound: 73.8395379
time: 6.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8395264, upper bound: 73.8395379
time: 6.53 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8396614, upper bound: 73.8396687
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8396615, upper bound: 73.8396680
time: 6.08 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8392965, upper bound: 73.8392869
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8392965, upper bound: 73.8392869
time: 9.15 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8401705, upper bound: 73.8401697
time: 6.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8401705, upper bound: 73.8401697
time: 5.47 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8439205, upper bound: 73.8439208
time: 7.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8439222, upper bound: 73.8439206
time: 6.91 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8360584, upper bound: 73.8360600
time: 6.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8360584, upper bound: 73.8360600
time: 7.83 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8446790, upper bound: 73.8446772
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8446790, upper bound: 73.8446772
time: 6.88 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8469834, upper bound: 73.8469855
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8469843, upper bound: 73.8469844
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8468225, upper bound: 73.8468248
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8468225, upper bound: 73.8468248
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8385178, upper bound: 73.8385164
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8385179, upper bound: 73.8385162
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8330429, upper bound: 73.8330442
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8330430, upper bound: 73.8330442
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8418305, upper bound: 73.8418338
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8418338, upper bound: 73.8418301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8371311, upper bound: 73.8371295
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8371311, upper bound: 73.8371295
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8477915, upper bound: 73.8477930
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8477914, upper bound: 73.8477940
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8477939, upper bound: 73.8477901
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8477937, upper bound: 73.8477908
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8395267, upper bound: 73.8395370
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8395293, upper bound: 73.8395339
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8395265, upper bound: 73.8395379
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8395264, upper bound: 73.8395379
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8396614, upper bound: 73.8396687
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8396615, upper bound: 73.8396680
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8392965, upper bound: 73.8392869
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8392965, upper bound: 73.8392869
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8401705, upper bound: 73.8401697
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8401705, upper bound: 73.8401697
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8439205, upper bound: 73.8439208
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8439222, upper bound: 73.8439206
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8360584, upper bound: 73.8360600
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8360584, upper bound: 73.8360600
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8446790, upper bound: 73.8446772
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 9, lower bound: -73.8446790, upper bound: 73.8446772

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8353693, upper bound: 73.8353711
time: 6.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8353693, upper bound: 73.8353711
time: 6.33 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8426765, upper bound: 73.8426757
time: 7.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8426777, upper bound: 73.8426738
time: 5.39 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8325411, upper bound: 73.8325401
time: 5.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8325411, upper bound: 73.8325401
time: 5.82 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8304739, upper bound: 73.8304722
time: 6.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8304739, upper bound: 73.8304722
time: 6.19 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8301510, upper bound: 73.8301449
time: 7.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8301510, upper bound: 73.8301449
time: 7.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 72

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8385164, upper bound: 73.8385162
time: 7.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8385179, upper bound: 73.8385150
time: 5.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8330429, upper bound: 73.8330435
time: 7.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8330428, upper bound: 73.8330443
time: 7.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8323145, upper bound: 73.8323152
time: 6.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -73.8323144, upper bound: 73.8323150
time: 7.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 232

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.35 + 589.96 = 601.31 seconds
