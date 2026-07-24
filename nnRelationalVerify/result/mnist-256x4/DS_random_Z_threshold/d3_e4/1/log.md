## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 173.89956106530002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329)
1: (-79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183)
2: (-104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471)
3: (-110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219)
4: (-101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509)
5: (-90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867)
6: (-86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223)
7: (-95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773)
8: (-114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580)
9: (-86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.86 + 10.24 = 11.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094897
time: 6.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094897
time: 6.10 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.29 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.29
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094897
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.29
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094897

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094888, upper bound: 174.0094897
time: 7.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094888
time: 7.13 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094896, upper bound: 174.0094897
time: 6.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094896
time: 6.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.41 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -174.0094888, upper bound: 174.0094897
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094888
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -174.0094896, upper bound: 174.0094897
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.41
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094896

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094888, upper bound: 174.0094840
time: 6.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094830, upper bound: 174.0094897
time: 7.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094876, upper bound: 174.0094888
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094875
time: 6.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015395, upper bound: 174.0015399
time: 6.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015371, upper bound: 174.0015406
time: 7.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034289
time: 6.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034289
time: 6.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.15 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0094888, upper bound: 174.0094840
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0094830, upper bound: 174.0094897
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0094876, upper bound: 174.0094888
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094875
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0015395, upper bound: 174.0015399
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0015371, upper bound: 174.0015406
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034289
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.15
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034289

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0067711, upper bound: 174.0067711
time: 7.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0067738, upper bound: 174.0067690
time: 6.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094830, upper bound: 174.0094864
time: 6.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094802, upper bound: 174.0094897
time: 6.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0072566, upper bound: 174.0072583
time: 6.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0072528, upper bound: 174.0072602
time: 6.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094868
time: 7.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094892, upper bound: 174.0094875
time: 7.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015394, upper bound: 174.0015399
time: 7.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015395, upper bound: 174.0015392
time: 6.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015371, upper bound: 174.0015262
time: 7.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015219, upper bound: 174.0015406
time: 6.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034258
time: 7.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034309, upper bound: 174.0034289
time: 7.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034315, upper bound: 174.0034289
time: 6.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034289
time: 7.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0067711, upper bound: 174.0067711
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0067738, upper bound: 174.0067690
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0094830, upper bound: 174.0094864
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0094802, upper bound: 174.0094897
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0072566, upper bound: 174.0072583
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0072528, upper bound: 174.0072602
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094868
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0094892, upper bound: 174.0094875
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0015394, upper bound: 174.0015399
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0015395, upper bound: 174.0015392
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0015371, upper bound: 174.0015262
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0015219, upper bound: 174.0015406
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034258
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0034309, upper bound: 174.0034289
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0034315, upper bound: 174.0034289
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.72
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034289

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 253

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011512, upper bound: 174.0011527
time: 6.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011512, upper bound: 174.0011527
time: 6.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926929
time: 8.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926929
time: 8.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034289, upper bound: 174.0034315
time: 6.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034289, upper bound: 174.0034315
time: 6.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 68

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9566223, upper bound: 173.9566270
time: 6.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9566223, upper bound: 173.9566270
time: 6.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0072566, upper bound: 174.0072526
time: 6.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0072518, upper bound: 174.0072583
time: 6.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0072512, upper bound: 174.0072602
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0072528, upper bound: 174.0072567
time: 6.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094810
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0094840, upper bound: 174.0094868
time: 6.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 91

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0002240, upper bound: 174.0002240
time: 6.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0002240, upper bound: 174.0002240
time: 7.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9991528, upper bound: 173.9991519
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9991507, upper bound: 173.9991530
time: 6.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9878766, upper bound: 173.9878800
time: 6.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9878766, upper bound: 173.9878800
time: 6.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015371, upper bound: 174.0015186
time: 6.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0015279, upper bound: 174.0015262
time: 6.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 126

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9786955, upper bound: 173.9787109
time: 6.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9786955, upper bound: 173.9787109
time: 6.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034233
time: 6.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034293, upper bound: 174.0034258
time: 6.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0006275, upper bound: 174.0006404
time: 5.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0006428, upper bound: 174.0006314
time: 6.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 226

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034315, upper bound: 174.0034032
time: 6.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034037, upper bound: 174.0034289
time: 6.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034270
time: 7.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034322, upper bound: 174.0034289
time: 7.30 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0011512, upper bound: 174.0011527
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0011512, upper bound: 174.0011527
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926929
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926929
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034289, upper bound: 174.0034315
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034289, upper bound: 174.0034315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9566223, upper bound: 173.9566270
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9566223, upper bound: 173.9566270
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0072566, upper bound: 174.0072526
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0072518, upper bound: 174.0072583
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0072512, upper bound: 174.0072602
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0072528, upper bound: 174.0072567
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094810
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0094840, upper bound: 174.0094868
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0002240, upper bound: 174.0002240
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0002240, upper bound: 174.0002240
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9991528, upper bound: 173.9991519
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9991507, upper bound: 173.9991530
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9878766, upper bound: 173.9878800
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9878766, upper bound: 173.9878800
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0015371, upper bound: 174.0015186
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0015279, upper bound: 174.0015262
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9786955, upper bound: 173.9787109
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -173.9786955, upper bound: 173.9787109
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034293, upper bound: 174.0034258
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0006275, upper bound: 174.0006404
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0006428, upper bound: 174.0006314
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034315, upper bound: 174.0034032
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034037, upper bound: 174.0034289
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034270
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.24
Output dim: 7, lower bound: -174.0034322, upper bound: 174.0034289

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011455, upper bound: 174.0011464
time: 6.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011448, upper bound: 174.0011470
time: 7.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Candidate
type: DSZ, layer: 1, pos: 217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011071, upper bound: 174.0011074
time: 6.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0011071, upper bound: 174.0011074
time: 6.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926923
time: 6.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926929
time: 6.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9858772, upper bound: 173.9858776
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9858742, upper bound: 173.9858798
time: 6.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939768, upper bound: 173.9939871
time: 6.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9939768, upper bound: 173.9939871
time: 6.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034289, upper bound: 174.0034302
time: 6.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0034265, upper bound: 174.0034315
time: 7.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9566217, upper bound: 173.9566270
time: 5.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9566223, upper bound: 173.9566265
time: 5.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -94.5060349, 75.0702057, -94.5060349, 75.0702057, -169.5762329, 169.5762329
1: -79.2014389, 66.5985794, -79.2014389, 66.5985794, -145.8000183, 145.8000183
2: -104.3030472, 68.0764999, -104.3030472, 68.0764999, -172.3795471, 172.3795471
3: -110.6649246, 58.1981163, -110.6649246, 58.1981163, -168.8630219, 168.8630219
4: -101.0963440, 77.7846146, -101.0963440, 77.7846146, -178.8809509, 178.8809509
5: -90.6905060, 70.5433807, -90.6905060, 70.5433807, -161.2338867, 161.2338867
6: -86.9384842, 83.7556839, -86.9384842, 83.7556839, -170.6941223, 170.6941223
7: -95.1351624, 80.1866226, -95.1351624, 80.1866226, -175.3217773, 175.3217773
8: -114.4460297, 77.5040588, -114.4460297, 77.5040588, -191.9500580, 191.9500580
9: -86.7146835, 84.7555695, -86.7146835, 84.7555695, -171.4702454, 171.4702454

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9519136, upper bound: 173.9519141
time: 6.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -173.9519087, upper bound: 173.9519169
time: 5.90 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 13.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -174.0011455, upper bound: 174.0011464
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -174.0011448, upper bound: 174.0011470
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -174.0011071, upper bound: 174.0011074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -174.0011071, upper bound: 174.0011074
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926923
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9926898, upper bound: 173.9926929
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9858772, upper bound: 173.9858776
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9858742, upper bound: 173.9858798
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9939768, upper bound: 173.9939871
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9939768, upper bound: 173.9939871
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -174.0034289, upper bound: 174.0034302
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -174.0034265, upper bound: 174.0034315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9566217, upper bound: 173.9566270
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9566223, upper bound: 173.9566265
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9519136, upper bound: 173.9519141
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.32
Output dim: 7, lower bound: -173.9519087, upper bound: 173.9519169
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0072566, upper bound: 174.0072526
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0072518, upper bound: 174.0072583
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0072512, upper bound: 174.0072602
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0072528, upper bound: 174.0072567
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0094897, upper bound: 174.0094810
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0094840, upper bound: 174.0094868
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0002240, upper bound: 174.0002240
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0002240, upper bound: 174.0002240
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -173.9991528, upper bound: 173.9991519
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -173.9991507, upper bound: 173.9991530
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -173.9878766, upper bound: 173.9878800
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -173.9878766, upper bound: 173.9878800
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0015371, upper bound: 174.0015186
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0015279, upper bound: 174.0015262
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -173.9786955, upper bound: 173.9787109
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -173.9786955, upper bound: 173.9787109
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0034293, upper bound: 174.0034258
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0006275, upper bound: 174.0006404
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0006428, upper bound: 174.0006314
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0034315, upper bound: 174.0034032
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0034037, upper bound: 174.0034289
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0034326, upper bound: 174.0034270
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.32
Output dim: 7, lower bound: -174.0034322, upper bound: 174.0034289

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.10 + 590.41 = 601.51 seconds
