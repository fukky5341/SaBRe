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
execution time: IAR + RelationalAnalysis = 0.80 + 10.22 = 11.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -174.0736347, upper bound: 174.0736347

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
time: 7.86 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
time: 7.32 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.25 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.25
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.25
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484527

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484529
time: 6.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458
time: 7.21 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484527
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458
time: 6.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.83 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.83
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484529
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.83
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.83
Output dim: 7, lower bound: -174.0484460, upper bound: 174.0484527
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.83
Output dim: 7, lower bound: -174.0484529, upper bound: 174.0484458

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
time: 7.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
time: 7.87 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 5.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 6.44 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
time: 7.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
time: 7.58 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 5.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
time: 6.47 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 13.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478808
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478689, upper bound: 174.0478807
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 13.10
Output dim: 7, lower bound: -174.0478808, upper bound: 174.0478688

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 6.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 6.56 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.25 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.62 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.57 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.72 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
time: 7.10 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.66 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
time: 6.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 14.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047400, upper bound: 174.0047437
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 14.10
Output dim: 7, lower bound: -174.0047437, upper bound: 174.0047400

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 6.91 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 6.99 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 6.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 6.83 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 6.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 6.59 seconds

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
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.35 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 6.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.32 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.39 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.30 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.11 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.10 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.83 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
time: 7.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
time: 7.79 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.04 seconds

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
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.16 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 7.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.31 seconds

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

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
time: 8.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
time: 7.49 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 16.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.91
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
time: 6.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
time: 6.21 seconds

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
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
time: 6.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
time: 6.16 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
time: 6.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
time: 6.20 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
time: 6.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
time: 6.08 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
time: 6.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
time: 6.04 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
time: 6.12 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
time: 6.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
time: 6.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 255
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021810, upper bound: 174.0021793
time: 7.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -174.0021841, upper bound: 174.0021773
time: 6.31 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 14.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021795, upper bound: 174.0021807
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021824, upper bound: 174.0021786
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021773, upper bound: 174.0021841
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021793, upper bound: 174.0021810
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021810, upper bound: 174.0021793
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.27
Output dim: 7, lower bound: -174.0021841, upper bound: 174.0021773
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047333, upper bound: 174.0047364
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047323, upper bound: 174.0047370
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047370, upper bound: 174.0047323
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.27
Output dim: 7, lower bound: -174.0047364, upper bound: 174.0047333

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.02 + 594.12 = 605.14 seconds
