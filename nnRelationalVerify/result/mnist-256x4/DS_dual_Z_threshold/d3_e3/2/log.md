## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 27.4911861951


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.20 + 15.23 = 17.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187042

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184961
time: 6.21 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184969
time: 7.35 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.82 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.82
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184961
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.82
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184969

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184941
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184948, upper bound: 27.5184963
time: 20.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184944
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5184945, upper bound: 27.5184965
time: 10.69 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 18.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.67
Output dim: 1, lower bound: -27.5184969, upper bound: 27.5184941
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.67
Output dim: 1, lower bound: -27.5184948, upper bound: 27.5184963
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 18.67
Output dim: 1, lower bound: -27.5184965, upper bound: 27.5184944
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 18.67
Output dim: 1, lower bound: -27.5184945, upper bound: 27.5184965

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109456
time: 10.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
time: 7.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
time: 9.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109461
time: 7.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
time: 6.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
time: 6.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
time: 16.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
time: 5.74 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 24.14 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109456
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109461
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109463, upper bound: 27.5109458
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.14
Output dim: 1, lower bound: -27.5109458, upper bound: 27.5109463

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
time: 7.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 6.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 6.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 7.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
time: 8.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
time: 7.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
time: 9.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
time: 7.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
time: 8.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
time: 4.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086428
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
time: 4.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
time: 7.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
time: 6.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
time: 20.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
time: 10.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086429
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086429
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086428
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086430, upper bound: 27.5086430
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086430
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.06
Output dim: 1, lower bound: -27.5086429, upper bound: 27.5086428

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 10.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 7.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 9.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083894
time: 8.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 10.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 5.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
time: 9.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083893
time: 8.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893
time: 6.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083896
time: 8.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083896
time: 8.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 7.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
time: 9.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083895
time: 7.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
time: 8.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 6.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
time: 7.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
time: 10.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
time: 7.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893
time: 6.45 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083894
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083895
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083893
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083893, upper bound: 27.5083896
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083896
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083895, upper bound: 27.5083895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083894
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083894, upper bound: 27.5083895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083894
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 1, lower bound: -27.5083896, upper bound: 27.5083893

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 17.44 + 584.11 = 601.54 seconds
