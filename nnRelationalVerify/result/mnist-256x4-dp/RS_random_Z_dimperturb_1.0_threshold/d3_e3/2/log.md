## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 27.4911861951


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255)
1: (-20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821)
2: (-25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841)
3: (-29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502)
4: (-27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682)
5: (-20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750)
6: (-22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550)
7: (-26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649)
8: (-32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634)
9: (-19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 15.75 = 17.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187042

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5187048, upper bound: 27.5187045
time: 9.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5187048, upper bound: 27.5187043
time: 8.49 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.56
Output dim: 1, lower bound: -27.5187048, upper bound: 27.5187045
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.56
Output dim: 1, lower bound: -27.5187048, upper bound: 27.5187043

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5065302, upper bound: 27.5065301
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5065302, upper bound: 27.5065301
time: 5.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125846
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125844
time: 9.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.97 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 1, lower bound: -27.5065302, upper bound: 27.5065301
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 1, lower bound: -27.5065302, upper bound: 27.5065301
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125846
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125844

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5064435, upper bound: 27.5064435
time: 7.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5064435, upper bound: 27.5064435
time: 9.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990520, upper bound: 27.4990521
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990520, upper bound: 27.4990521
time: 16.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5124401, upper bound: 27.5124400
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5124404, upper bound: 27.5124400
time: 8.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125845
time: 7.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125844
time: 6.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.5064435, upper bound: 27.5064435
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.5064435, upper bound: 27.5064435
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.4990520, upper bound: 27.4990521
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.4990520, upper bound: 27.4990521
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.5124401, upper bound: 27.5124400
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.5124404, upper bound: 27.5124400
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125845
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.51
Output dim: 1, lower bound: -27.5125846, upper bound: 27.5125844

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5041390, upper bound: 27.5041398
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5041390, upper bound: 27.5041398
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5062299, upper bound: 27.5062303
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5062300, upper bound: 27.5062303
time: 9.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990520, upper bound: 27.4990518
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990517, upper bound: 27.4990521
time: 7.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990430, upper bound: 27.4990430
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990431
time: 9.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5097993, upper bound: 27.5098000
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5097993, upper bound: 27.5098000
time: 5.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 112

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5124403, upper bound: 27.5124400
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5124404, upper bound: 27.5124399
time: 33.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5122169, upper bound: 27.5122169
time: 7.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5122169, upper bound: 27.5122169
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5114880, upper bound: 27.5114886
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5114883, upper bound: 27.5114887
time: 7.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5041390, upper bound: 27.5041398
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5041390, upper bound: 27.5041398
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5062299, upper bound: 27.5062303
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5062300, upper bound: 27.5062303
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.4990520, upper bound: 27.4990518
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.4990517, upper bound: 27.4990521
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.4990430, upper bound: 27.4990430
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990431
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5097993, upper bound: 27.5098000
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5097993, upper bound: 27.5098000
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5124403, upper bound: 27.5124400
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5124404, upper bound: 27.5124399
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5122169, upper bound: 27.5122169
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5122169, upper bound: 27.5122169
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5114880, upper bound: 27.5114886
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.59
Output dim: 1, lower bound: -27.5114883, upper bound: 27.5114887

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4999994, upper bound: 27.4999996
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4999994, upper bound: 27.4999996
time: 9.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5038595, upper bound: 27.5038603
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5038592, upper bound: 27.5038605
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5017887, upper bound: 27.5017885
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5017887, upper bound: 27.5017885
time: 6.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5058939, upper bound: 27.5058937
time: 19.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5058940, upper bound: 27.5058939
time: 7.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4973691, upper bound: 27.4973687
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4973691, upper bound: 27.4973691
time: 7.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4973692, upper bound: 27.4973687
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4973692, upper bound: 27.4973683
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4989409, upper bound: 27.4989415
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4989414, upper bound: 27.4989409
time: 8.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990431
time: 7.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990430
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5014736, upper bound: 27.5014738
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5014736, upper bound: 27.5014738
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086007, upper bound: 27.5086012
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086006, upper bound: 27.5086010
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 221

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5124404, upper bound: 27.5124394
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5124399, upper bound: 27.5124403
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 156

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5113127, upper bound: 27.5113138
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5113146, upper bound: 27.5113119
time: 5.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 198

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5098084, upper bound: 27.5098079
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5098085, upper bound: 27.5098080
time: 9.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5025262, upper bound: 27.5025258
time: 15.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5025262, upper bound: 27.5025258
time: 15.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5114769, upper bound: 27.5114774
time: 9.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5114769, upper bound: 27.5114773
time: 13.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 119

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4963019, upper bound: 27.4963014
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4963019, upper bound: 27.4963008
time: 13.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4999994, upper bound: 27.4999996
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4999994, upper bound: 27.4999996
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5038595, upper bound: 27.5038603
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5038592, upper bound: 27.5038605
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5017887, upper bound: 27.5017885
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5017887, upper bound: 27.5017885
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5058939, upper bound: 27.5058937
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5058940, upper bound: 27.5058939
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4973691, upper bound: 27.4973687
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4973691, upper bound: 27.4973691
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4973692, upper bound: 27.4973687
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4973692, upper bound: 27.4973683
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4989409, upper bound: 27.4989415
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4989414, upper bound: 27.4989409
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990430
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5014736, upper bound: 27.5014738
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5014736, upper bound: 27.5014738
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5086007, upper bound: 27.5086012
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5086006, upper bound: 27.5086010
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5124404, upper bound: 27.5124394
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5124399, upper bound: 27.5124403
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5113127, upper bound: 27.5113138
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5113146, upper bound: 27.5113119
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5098084, upper bound: 27.5098079
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5098085, upper bound: 27.5098080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5025262, upper bound: 27.5025258
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5025262, upper bound: 27.5025258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5114769, upper bound: 27.5114774
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.5114769, upper bound: 27.5114773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4963019, upper bound: 27.4963014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.33
Output dim: 1, lower bound: -27.4963019, upper bound: 27.4963008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.4847088, 19.3536148, -21.4847088, 19.3536148, -40.8383255, 40.8383255
1: -20.9076710, 13.3137093, -20.9076710, 13.3137093, -34.2213783, 34.2213821
2: -25.2017784, 16.5133114, -25.2017784, 16.5133114, -41.7150841, 41.7150841
3: -29.8778210, 14.4127302, -29.8778210, 14.4127302, -44.2905502, 44.2905502
4: -27.2308311, 17.5701427, -27.2308311, 17.5701427, -44.8009682, 44.8009682
5: -20.8884811, 18.7889977, -20.8884811, 18.7889977, -39.6774750, 39.6774750
6: -22.1692429, 20.1599159, -22.1692429, 20.1599159, -42.3291550, 42.3291550
7: -26.5594749, 19.4237938, -26.5594749, 19.4237938, -45.9832649, 45.9832649
8: -32.1201401, 16.2826233, -32.1201401, 16.2826233, -48.4027634, 48.4027634
9: -19.4115181, 21.8075218, -19.4115181, 21.8075218, -41.2190247, 41.2190323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=56, inp2_unstable=56, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=123, inp2_unstable=123, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 130

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4999994, upper bound: 27.4999994
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4999991, upper bound: 27.4999996
time: 5.11 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 11.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.13
Output dim: 1, lower bound: -27.4999994, upper bound: 27.4999994
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.13
Output dim: 1, lower bound: -27.4999991, upper bound: 27.4999996
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4999994, upper bound: 27.4999996
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5038595, upper bound: 27.5038603
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5038592, upper bound: 27.5038605
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5017887, upper bound: 27.5017885
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5017887, upper bound: 27.5017885
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5058939, upper bound: 27.5058937
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5058940, upper bound: 27.5058939
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4973691, upper bound: 27.4973687
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4973691, upper bound: 27.4973691
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4973692, upper bound: 27.4973687
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4973692, upper bound: 27.4973683
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4989409, upper bound: 27.4989415
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4989414, upper bound: 27.4989409
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4990429, upper bound: 27.4990430
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5014736, upper bound: 27.5014738
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5014736, upper bound: 27.5014738
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5086007, upper bound: 27.5086012
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5086006, upper bound: 27.5086010
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5124404, upper bound: 27.5124394
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5124399, upper bound: 27.5124403
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5113127, upper bound: 27.5113138
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5113146, upper bound: 27.5113119
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5098084, upper bound: 27.5098079
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5098085, upper bound: 27.5098080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5025262, upper bound: 27.5025258
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5025262, upper bound: 27.5025258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5114769, upper bound: 27.5114774
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.5114769, upper bound: 27.5114773
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4963019, upper bound: 27.4963014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.13
Output dim: 1, lower bound: -27.4963019, upper bound: 27.4963008

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 17.24 + 586.38 = 603.62 seconds
