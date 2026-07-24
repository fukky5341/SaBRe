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
execution time: IAR + RelationalAnalysis = 0.88 + 15.51 = 16.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -27.5187049, upper bound: 27.5187042

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 208

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186010
time: 9.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186007
time: 8.63 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 17.87 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 17.87
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186010
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 17.87
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186007

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5185595, upper bound: 27.5185591
time: 6.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5185594, upper bound: 27.5185591
time: 5.73 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186008
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186011
time: 7.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.57 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.57
Output dim: 1, lower bound: -27.5185595, upper bound: 27.5185591
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.57
Output dim: 1, lower bound: -27.5185594, upper bound: 27.5185591
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.57
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186008
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.57
Output dim: 1, lower bound: -27.5186013, upper bound: 27.5186011

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5112446, upper bound: 27.5112440
time: 5.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5112446, upper bound: 27.5112440
time: 6.07 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5158434, upper bound: 27.5158432
time: 8.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5158434, upper bound: 27.5158433
time: 9.85 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 162

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5121766, upper bound: 27.5121769
time: 6.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5121766, upper bound: 27.5121772
time: 6.29 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5185612, upper bound: 27.5185604
time: 7.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5185613, upper bound: 27.5185610
time: 6.44 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.72 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5112446, upper bound: 27.5112440
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5112446, upper bound: 27.5112440
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5158434, upper bound: 27.5158432
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5158434, upper bound: 27.5158433
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5121766, upper bound: 27.5121769
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5121766, upper bound: 27.5121772
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5185612, upper bound: 27.5185604
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.72
Output dim: 1, lower bound: -27.5185613, upper bound: 27.5185610

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110005
time: 9.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110009, upper bound: 27.5110000
time: 8.07 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 234

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5023411, upper bound: 27.5023413
time: 6.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5023411, upper bound: 27.5023413
time: 6.90 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 212

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5085737, upper bound: 27.5085727
time: 7.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5085738, upper bound: 27.5085727
time: 5.64 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5134420, upper bound: 27.5134426
time: 10.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5134420, upper bound: 27.5134425
time: 30.14 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5100951, upper bound: 27.5100989
time: 8.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5100966, upper bound: 27.5100964
time: 5.66 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5057645, upper bound: 27.5057646
time: 8.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5057645, upper bound: 27.5057646
time: 8.01 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5182558, upper bound: 27.5182553
time: 7.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5182555, upper bound: 27.5182559
time: 5.93 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036197, upper bound: 27.5036190
time: 6.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036197, upper bound: 27.5036191
time: 6.76 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110005
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5110009, upper bound: 27.5110000
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5023411, upper bound: 27.5023413
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5023411, upper bound: 27.5023413
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5085737, upper bound: 27.5085727
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5085738, upper bound: 27.5085727
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5134420, upper bound: 27.5134426
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5134420, upper bound: 27.5134425
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5100951, upper bound: 27.5100989
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5100966, upper bound: 27.5100964
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5057645, upper bound: 27.5057646
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5057645, upper bound: 27.5057646
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5182558, upper bound: 27.5182553
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5182555, upper bound: 27.5182559
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5036197, upper bound: 27.5036190
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.77
Output dim: 1, lower bound: -27.5036197, upper bound: 27.5036191

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 59

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110003
time: 8.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110006, upper bound: 27.5110004
time: 8.42 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5038780, upper bound: 27.5038781
time: 8.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5038779, upper bound: 27.5038781
time: 7.48 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5000711, upper bound: 27.5000696
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5000711, upper bound: 27.5000712
time: 7.96 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5023331, upper bound: 27.5023351
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5023349, upper bound: 27.5023334
time: 7.98 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 112

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4965847, upper bound: 27.4965837
time: 5.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.4965847, upper bound: 27.4965837
time: 5.84 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5057683, upper bound: 27.5057672
time: 7.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5057683, upper bound: 27.5057672
time: 7.64 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5123280, upper bound: 27.5123300
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5123293, upper bound: 27.5123295
time: 9.88 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5091551, upper bound: 27.5091543
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5091551, upper bound: 27.5091544
time: 4.71 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 198

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5079529, upper bound: 27.5079574
time: 38.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5079529, upper bound: 27.5079575
time: 11.14 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 127

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036025, upper bound: 27.5036033
time: 9.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036025, upper bound: 27.5036033
time: 7.96 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5028629, upper bound: 27.5028667
time: 8.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5028637, upper bound: 27.5028654
time: 5.39 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -27.4877839, upper bound: 27.4877839
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -27.4877839, upper bound: 27.4877841
time: 11.61 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5181060, upper bound: 27.5181073
time: 7.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5181070, upper bound: 27.5181065
time: 9.05 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5149445, upper bound: 27.5149451
time: 11.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5149445, upper bound: 27.5149451
time: 9.56 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 130

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036195, upper bound: 27.5036189
time: 6.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036193, upper bound: 27.5036192
time: 9.01 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036066, upper bound: 27.5036056
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5036065, upper bound: 27.5036056
time: 5.97 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110003
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5110006, upper bound: 27.5110004
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5038780, upper bound: 27.5038781
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5038779, upper bound: 27.5038781
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5000711, upper bound: 27.5000696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5000711, upper bound: 27.5000712
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5023331, upper bound: 27.5023351
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5023349, upper bound: 27.5023334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.4965847, upper bound: 27.4965837
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.4965847, upper bound: 27.4965837
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5057683, upper bound: 27.5057672
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5057683, upper bound: 27.5057672
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5123280, upper bound: 27.5123300
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5123293, upper bound: 27.5123295
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5091551, upper bound: 27.5091543
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5091551, upper bound: 27.5091544
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5079529, upper bound: 27.5079574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5079529, upper bound: 27.5079575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5036025, upper bound: 27.5036033
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5036025, upper bound: 27.5036033
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5028629, upper bound: 27.5028667
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5028637, upper bound: 27.5028654
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.4877839, upper bound: 27.4877839
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.4877839, upper bound: 27.4877841
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5181060, upper bound: 27.5181073
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5181070, upper bound: 27.5181065
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5149445, upper bound: 27.5149451
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5149445, upper bound: 27.5149451
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5036195, upper bound: 27.5036189
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5036193, upper bound: 27.5036192
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5036066, upper bound: 27.5036056
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.48
Output dim: 1, lower bound: -27.5036065, upper bound: 27.5036056

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 59

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110003
time: 7.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110004
time: 8.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110000, upper bound: 27.5110004
time: 5.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5109999
time: 6.58 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 12.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.65
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110003
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.65
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5110004
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.65
Output dim: 1, lower bound: -27.5110000, upper bound: 27.5110004
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.65
Output dim: 1, lower bound: -27.5110007, upper bound: 27.5109999
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5038780, upper bound: 27.5038781
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5038779, upper bound: 27.5038781
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5000711, upper bound: 27.5000696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5000711, upper bound: 27.5000712
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5023331, upper bound: 27.5023351
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5023349, upper bound: 27.5023334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.4965847, upper bound: 27.4965837
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.4965847, upper bound: 27.4965837
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5057683, upper bound: 27.5057672
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5057683, upper bound: 27.5057672
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5123280, upper bound: 27.5123300
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5123293, upper bound: 27.5123295
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5091551, upper bound: 27.5091543
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5091551, upper bound: 27.5091544
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5079529, upper bound: 27.5079574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5079529, upper bound: 27.5079575
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5036025, upper bound: 27.5036033
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5036025, upper bound: 27.5036033
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5028629, upper bound: 27.5028667
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5028637, upper bound: 27.5028654
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5181060, upper bound: 27.5181073
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5181070, upper bound: 27.5181065
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5149445, upper bound: 27.5149451
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5149445, upper bound: 27.5149451
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5036195, upper bound: 27.5036189
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5036193, upper bound: 27.5036192
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5036066, upper bound: 27.5036056
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.65
Output dim: 1, lower bound: -27.5036065, upper bound: 27.5036056

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 16.39 + 585.71 = 602.11 seconds
