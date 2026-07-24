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
execution time: IAR + RelationalAnalysis = 1.49 + 15.64 = 17.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187042

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184961
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184969
time: 7.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.01 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.01
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184961
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.01
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184969

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184941
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184948, upper bound: 27.5184963
time: 21.43 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184944
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184945, upper bound: 27.5184965
time: 11.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 18.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.38
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184941
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.38
Output dim: 1, lower bound: -27.5184948, upper bound: 27.5184963
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 18.38
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184944
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 18.38
Output dim: 1, lower bound: -27.5184945, upper bound: 27.5184965

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109456
time: 10.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
time: 7.58 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
time: 9.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109461
time: 7.49 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
time: 6.27 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
time: 16.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
time: 5.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109456
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109461
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
time: 7.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 6.84 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 7.36 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
time: 7.76 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
time: 10.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
time: 7.27 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 4.95 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086428
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
time: 5.05 seconds

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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
time: 7.43 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
time: 22.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
time: 10.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 34.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086428
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.51
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 10.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 7.63 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 9.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083894
time: 8.30 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 10.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.03 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 9.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.33 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083893
time: 8.82 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 8.03 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 8.24 seconds

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893
time: 6.20 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083896
time: 8.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.27 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.83 seconds

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083896
time: 8.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 7.34 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.41 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083895
time: 8.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
time: 8.60 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.84 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 10.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
time: 7.68 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893
time: 6.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083894
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083893
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083896
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083896
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.03
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 221
type: RSZ, layer: 1, pos: 119
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 234
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 198
type: RSZ, layer: 1, pos: 162
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 56
type: RSZ, layer: 1, pos: 130
type: RSZ, layer: 1, pos: 156
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 59
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 212
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 112

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5021349, upper bound: 27.5021342
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5021349, upper bound: 27.5021344
time: 6.99 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 19.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 19.07
Output dim: 1, lower bound: -27.5021349, upper bound: 27.5021342
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 19.07
Output dim: 1, lower bound: -27.5021349, upper bound: 27.5021344
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083894
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083893
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083896
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083896
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.07
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 17.13 + 587.52 = 604.65 seconds
