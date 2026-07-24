## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 206.199692099
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-133.0105286, 105.4250412, -133.0105286, 105.4250412, -238.4355469, 238.4355469)
1: (-113.2106094, 94.0342941, -113.2106094, 94.0342941, -207.2449036, 207.2449036)
2: (-147.5505219, 95.8398438, -147.5505219, 95.8398438, -243.3903656, 243.3903656)
3: (-156.0251617, 82.8341141, -156.0251617, 82.8341141, -238.8592682, 238.8592682)
4: (-143.4313812, 109.8025894, -143.4313812, 109.8025894, -253.2339783, 253.2339783)
5: (-127.3003082, 99.2028809, -127.3003082, 99.2028809, -226.5031586, 226.5031586)
6: (-122.3499832, 118.8279724, -122.3499832, 118.8279724, -241.1779480, 241.1779480)
7: (-134.1060638, 112.8064041, -134.1060638, 112.8064041, -246.9124603, 246.9124603)
8: (-162.7028046, 111.3033218, -162.7028046, 111.3033218, -274.0060730, 274.0060730)
9: (-122.2542877, 119.9744720, -122.2542877, 119.9744720, -242.2287292, 242.2287292)

## BASE Result
execution time: IAR + LP analysis = 1.27 + 11.02 = 12.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -206.2537789, upper bound: 206.2537789


# Binary Search by BASE starts (time budget: 2687.71 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.24490356445312
rel_dist={1: [-206.25362701135504, 206.25362701135498]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.24490356445312
rel_dist={1: [-206.2534079160462, 206.25340791604617]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.24490356445312
rel_dist={1: [-206.25298865227455, 206.2529886525199]}

## Binary Search Result
Binary search time: 42.01 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2645.70 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2176327, upper bound: 206.2114031
time: 10.15 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2009961, upper bound: 206.2009961
time: 6.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.04
Output dim: 1, lower bound: -206.2176327, upper bound: 206.2114031
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.04
Output dim: 1, lower bound: -206.2009961, upper bound: 206.2009961

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -131.5043640, 104.2381363, -133.0105286, 105.4250412, -236.9293976, 237.2486420
1: -111.9491501, 92.9773102, -113.2106094, 94.0342941, -205.9834442, 206.1878967
2: -145.8954468, 94.7646484, -147.5505219, 95.8398438, -241.7352905, 242.3151398
3: -154.2395020, 81.9059525, -156.0251617, 82.8341141, -237.0735931, 237.9311218
4: -141.8089142, 108.5704422, -143.4313812, 109.8025894, -251.6115112, 252.0018311
5: -125.8522491, 98.0755768, -127.3003082, 99.2028809, -225.0551147, 225.3758850
6: -120.9647751, 117.4926453, -122.3499832, 118.8279724, -239.7927551, 239.8426208
7: -132.5888824, 111.5377655, -134.1060638, 112.8064041, -245.3952637, 245.6438293
8: -160.8848419, 110.0725784, -162.7028046, 111.3033218, -272.1881409, 272.7753296
9: -120.8762054, 118.6215515, -122.2542877, 119.9744720, -240.8506470, 240.8758087

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1925598, upper bound: 206.1897600
time: 9.67 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1884521, upper bound: 206.1870476
time: 9.23 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1715998, upper bound: 206.1700648
time: 9.36 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2125895, upper bound: 206.2068299
time: 9.71 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2176327, upper bound: 206.2114031
time: 9.50 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -124.2551422, 98.5684891, -131.3976746, 104.1528168, -228.4079590, 229.9661407
1: -105.9272766, 87.8787537, -111.8620682, 92.9014282, -198.8286896, 199.7408142
2: -137.9940186, 89.5577393, -145.7796326, 94.6904984, -232.6845093, 235.3373718
3: -145.5234070, 77.3668823, -154.1123199, 81.8394165, -227.3628235, 231.4792023
4: -133.9168549, 102.5689545, -141.6887512, 108.4823074, -242.3991699, 244.2577057
5: -118.8385925, 92.4406509, -125.7492294, 97.9922104, -216.8307648, 218.1898804
6: -114.3208389, 111.0633698, -120.8641586, 117.3971100, -231.7179565, 231.9275208
7: -125.2585983, 105.4193268, -132.4814911, 111.4492874, -236.7078857, 237.9007874
8: -152.1544647, 104.1155319, -160.7567902, 109.9849777, -262.1394348, 264.8722839
9: -114.1511459, 112.0845032, -120.7780609, 118.5239182, -232.6750488, 232.8625641

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1966314, upper bound: 206.1966913
time: 6.93 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1965656, upper bound: 206.1965656
time: 8.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 57.46 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 57.46
Output dim: 1, lower bound: -206.2125895, upper bound: 206.2068299
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 57.46
Output dim: 1, lower bound: -206.2176327, upper bound: 206.2114031
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 57.46
Output dim: 1, lower bound: -206.1966314, upper bound: 206.1966913
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 57.46
Output dim: 1, lower bound: -206.1965656, upper bound: 206.1965656

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -127.9563828, 101.4034119, -132.0014801, 104.6253891, -232.5817719, 233.4048767
1: -108.9914932, 90.4560242, -112.3648682, 93.3222733, -202.3137665, 202.8208923
2: -141.9002533, 92.2146225, -146.4296265, 95.1206741, -237.0209198, 238.6442566
3: -150.0061035, 79.7032776, -154.8329620, 82.2108688, -232.2169342, 234.5362396
4: -138.0894775, 105.6241608, -142.3510132, 108.9682159, -247.0576935, 247.9751740
5: -122.4302673, 95.4394608, -126.3305817, 98.4543991, -220.8846741, 221.7700348
6: -117.7061081, 114.3473206, -121.4192963, 117.9340057, -235.6401062, 235.7666168
7: -128.9674072, 108.4788132, -133.0853577, 111.9518890, -240.9192963, 241.5641479
8: -156.6314697, 107.1919098, -161.4878387, 110.4756393, -267.1071167, 268.6797485
9: -117.6271591, 115.4647675, -121.3305054, 119.0710297, -236.6981812, 236.7952423

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1684362, upper bound: 206.1664613
time: 10.69 seconds

## Relational analysis of IS_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2081640, upper bound: 206.2032588
time: 10.30 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2101724, upper bound: 206.2051574
time: 8.47 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -129.5218201, 102.6666107, -133.0105286, 105.4250412, -234.9468384, 235.6771088
1: -110.2867508, 91.5776291, -113.2106094, 94.0342941, -204.3210449, 204.7882385
2: -143.6921387, 93.3507538, -147.5505219, 95.8398438, -239.5319672, 240.9012756
3: -151.9043579, 80.6800385, -156.0251617, 82.8341141, -234.7384491, 236.7052002
4: -139.6912231, 106.9313049, -143.4313812, 109.8025894, -249.4938049, 250.3626862
5: -123.9523163, 96.6070557, -127.3003082, 99.2028809, -223.1551666, 223.9073639
6: -119.1373215, 115.7389755, -122.3499832, 118.8279724, -237.9652863, 238.0889587
7: -130.5812378, 109.8553543, -134.1060638, 112.8064041, -243.3876343, 243.9614105
8: -158.5032043, 108.4502411, -162.7028046, 111.3033218, -269.8065186, 271.1530151
9: -119.0599518, 116.8495941, -122.2542877, 119.9744720, -239.0344086, 239.1038818

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1925598, upper bound: 206.1897600
time: 10.42 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1884521, upper bound: 206.1870476
time: 10.25 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1715998, upper bound: 206.1700648
time: 9.71 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1818694, upper bound: 206.1820064
time: 9.82 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1614757, upper bound: 206.1619883
time: 9.81 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2146113, upper bound: 206.2091903
time: 10.15 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2162469, upper bound: 206.2106742
time: 9.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 125.15 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 125.15
Output dim: 1, lower bound: -206.2081640, upper bound: 206.2032588
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 125.15
Output dim: 1, lower bound: -206.2101724, upper bound: 206.2051574
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 125.15
Output dim: 1, lower bound: -206.2146113, upper bound: 206.2091903
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 125.15
Output dim: 1, lower bound: -206.2162469, upper bound: 206.2106742

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -127.5526505, 101.0840988, -121.3478165, 96.2022171, -223.7548676, 222.4319153
1: -108.6567078, 90.1736984, -103.5024872, 85.8717804, -194.5284882, 193.6761780
2: -141.4537354, 91.9275970, -134.6624756, 87.5532990, -229.0070190, 226.5900726
3: -149.5356445, 79.4572830, -142.4079285, 75.7007599, -225.2364044, 221.8652039
4: -137.6590118, 105.2958298, -130.9712677, 100.2933044, -237.9523163, 236.2670898
5: -122.0433426, 95.1399384, -116.1206665, 90.5569153, -212.6002197, 211.2606049
6: -117.3383026, 113.9901581, -111.6810989, 108.5040588, -225.8423462, 225.6712341
7: -128.5613708, 108.1390305, -122.3811569, 102.9990387, -231.5603943, 230.5201721
8: -156.1472015, 106.8640366, -148.7070618, 101.7931671, -257.9403687, 255.5711060
9: -117.2601242, 115.1053848, -111.6353302, 109.5588913, -226.8190155, 226.7406769

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1568905, upper bound: 206.1551890
time: 9.68 seconds

## Relational analysis of IS_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2081640, upper bound: 206.2032588
time: 9.54 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2081640, upper bound: 206.2032588
time: 9.26 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -127.8872452, 101.3487015, -124.7625580, 98.9005966, -226.7878418, 226.1112518
1: -108.9342957, 90.4077225, -106.3741989, 88.2666626, -197.2009583, 196.7819214
2: -141.8237762, 92.1655197, -138.4241791, 89.9770737, -231.8008423, 230.5896912
3: -149.9255981, 79.6612854, -146.3981476, 77.8208771, -227.7464752, 226.0594330
4: -138.0158539, 105.5679550, -134.6405182, 103.0861130, -241.1019592, 240.2084656
5: -122.3640900, 95.3881531, -119.4005585, 93.0794601, -215.4435425, 214.7887115
6: -117.6432571, 114.2862473, -114.8390350, 111.5437393, -229.1869965, 229.1252747
7: -128.8979034, 108.4206772, -125.8129959, 105.8696518, -234.7675476, 234.2336731
8: -156.5486755, 107.1357574, -152.8197174, 104.5965347, -261.1452026, 259.9554749
9: -117.5644226, 115.4032135, -114.7667313, 112.6252136, -230.1896362, 230.1699524

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1622205, upper bound: 206.1598284
time: 10.28 seconds

## Relational analysis of IS_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2100139, upper bound: 206.2050445
time: 8.57 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2100139, upper bound: 206.2051574
time: 9.72 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -129.1135864, 102.3437958, -122.3482819, 96.9952698, -226.1088409, 224.6920776
1: -109.9482727, 91.2921829, -104.3410645, 86.5778198, -196.5260773, 195.6332397
2: -143.2405243, 93.0605774, -135.7731781, 88.2661514, -231.5066528, 228.8337555
3: -151.4286346, 80.4313126, -143.5900574, 76.3189392, -227.7475739, 224.0213623
4: -139.2561646, 106.5993042, -132.0429077, 101.1205978, -240.3767700, 238.6421967
5: -123.5610962, 96.3043137, -117.0824127, 91.2991104, -214.8601685, 213.3867188
6: -118.7655945, 115.3778305, -112.6044006, 109.3904037, -228.1559906, 227.9822388
7: -130.1707153, 109.5117950, -123.3929977, 103.8461914, -234.0169067, 232.9047852
8: -158.0136566, 108.1187057, -149.9116974, 102.6137314, -260.6273804, 258.0303650
9: -118.6887512, 116.4862671, -112.5510635, 110.4543991, -229.1431580, 229.0373077

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1878413, upper bound: 206.1854421
time: 7.79 seconds

## Relational analysis of IS_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=207.24490356445312
rel_dist={1: [-206.25362701135504, 206.25362701135498]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2063172, upper bound: 206.2096691
time: 10.88 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2008875, upper bound: 206.2008875
time: 8.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.47 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 19.47
Output dim: 1, lower bound: -206.2063172, upper bound: 206.2096691
IS_B2, status: Status.UNKNOWN, split count: 1, time: 19.47
Output dim: 1, lower bound: -206.2008875, upper bound: 206.2008875

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -133.0105286, 105.4250412, -131.5043640, 104.2381363, -237.2486420, 236.9293976
1: -113.2106094, 94.0342941, -111.9491501, 92.9773102, -206.1878967, 205.9834442
2: -147.5505219, 95.8398438, -145.8954468, 94.7646484, -242.3151398, 241.7352905
3: -156.0251617, 82.8341141, -154.2395020, 81.9059525, -237.9311218, 237.0735931
4: -143.4313812, 109.8025894, -141.8089142, 108.5704422, -252.0018311, 251.6115112
5: -127.3003082, 99.2028809, -125.8522491, 98.0755768, -225.3758850, 225.0551147
6: -122.3499832, 118.8279724, -120.9647751, 117.4926453, -239.8426208, 239.7927551
7: -134.1060638, 112.8064041, -132.5888824, 111.5377655, -245.6438293, 245.3952637
8: -162.7028046, 111.3033218, -160.8848419, 110.0725784, -272.7753296, 272.1881409
9: -122.2542877, 119.9744720, -120.8762054, 118.6215515, -240.8758087, 240.8506470

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1621818, upper bound: 206.1623638
time: 9.21 seconds

## Relational analysis of IS_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1452585, upper bound: 206.1454960
time: 8.67 seconds

## Relational analysis of IS_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1549526, upper bound: 206.1548760
time: 8.50 seconds

## Relational analysis of IS_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2038683, upper bound: 206.2066994
time: 9.68 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2055577, upper bound: 206.2085874
time: 9.06 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -128.9001923, 102.1819077, -124.2551422, 98.5684891, -227.4686584, 226.4370422
1: -109.7738800, 91.1471329, -105.9272766, 87.8787537, -197.6526184, 197.0744019
2: -143.0373535, 92.9107437, -137.9940186, 89.5577393, -232.5950775, 230.9047546
3: -151.1499634, 80.2991867, -145.5234070, 77.3668823, -228.5168304, 225.8226013
4: -138.9898224, 106.4378204, -133.9168549, 102.5689545, -241.5587769, 240.3546753
5: -123.3467865, 96.1174850, -118.8385925, 92.4406509, -215.7874451, 214.9560242
6: -118.5631790, 115.1813126, -114.3208389, 111.0633698, -229.6265564, 229.5021515
7: -129.9664001, 109.3483124, -125.2585983, 105.4193268, -235.3856964, 234.6069031
8: -157.7431641, 107.9428406, -152.1544647, 104.1155319, -261.8587036, 260.0972900
9: -118.4920654, 116.2772141, -114.1511459, 112.0845032, -230.5765533, 230.4283295

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1982633, upper bound: 206.1983285
time: 7.30 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1999715, upper bound: 206.1999715
time: 7.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 53.09 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 53.09
Output dim: 1, lower bound: -206.2038683, upper bound: 206.2066994
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 53.09
Output dim: 1, lower bound: -206.2055577, upper bound: 206.2085874
IS_B2_A1, status: Status.VERIFIED, split count: 2, time: 53.09
Output dim: 1, lower bound: -206.1982633, upper bound: 206.1983285
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 53.09
Output dim: 1, lower bound: -206.1999715, upper bound: 206.1999715

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -122.3482819, 96.9952698, -128.2175903, 101.6387634, -223.9870453, 225.2128448
1: -104.3410645, 86.5778198, -109.2234039, 90.6792679, -195.0203247, 195.8012238
2: -135.7731781, 88.2661514, -142.2594147, 92.4280472, -228.2012329, 230.5255432
3: -143.5900574, 76.3189392, -150.4092255, 79.9035568, -223.4935913, 226.7281647
4: -132.0429077, 101.1205978, -138.3064423, 105.8974686, -237.9403687, 239.4270325
5: -117.0824127, 91.2991104, -122.7021942, 95.6380692, -212.7204742, 214.0013123
6: -112.6044006, 109.3904037, -117.9718094, 114.5851364, -227.1895447, 227.3622131
7: -123.3929977, 103.8461914, -129.2842407, 108.7717667, -232.1647644, 233.1304321
8: -149.9116974, 102.6137314, -156.9434509, 107.4033813, -257.3150635, 259.5571289
9: -112.5510635, 110.4543991, -117.8882370, 115.6957779, -228.2468414, 228.3426361

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2038596, upper bound: 206.2066994
time: 9.78 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2038596, upper bound: 206.2066994
time: 11.20 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -125.7769241, 99.7046509, -128.8566284, 102.1429367, -227.9198303, 228.5612793
1: -107.2242966, 88.9826508, -109.7570953, 91.1277618, -198.3520508, 198.7397156
2: -139.5509796, 90.6998215, -142.9661255, 92.8831177, -232.4340820, 233.6659241
3: -147.5962524, 78.4478607, -151.1544647, 80.2997131, -227.8959503, 229.6023254
4: -135.7267151, 103.9251862, -138.9888611, 106.4182739, -242.1449738, 242.9140472
5: -120.3754959, 93.8318710, -123.3164291, 96.1097107, -216.4851837, 217.1483002
6: -115.7748718, 112.4425888, -118.5574570, 115.1543427, -230.9291840, 231.0000458
7: -126.8394623, 106.7288284, -129.9280853, 109.3117218, -236.1511841, 236.6569061
8: -154.0409698, 105.4287491, -157.7134552, 107.9224243, -261.9633789, 263.1422119
9: -115.6958160, 113.5333557, -118.4740677, 116.2634048, -231.9591827, 232.0074158

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2049637, upper bound: 206.2079792
time: 11.68 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2049637, upper bound: 206.2085872
time: 9.33 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -121.6883545, 96.4792099, -121.6874924, 96.5365524, -218.2248993, 218.1667023
1: -103.8057480, 86.1105194, -103.8014069, 86.0842743, -189.8900146, 189.9119263
2: -135.0619812, 87.7852859, -135.1510315, 87.7321701, -222.7941284, 222.9363098
3: -142.7464905, 75.9260101, -142.5329285, 75.8052673, -218.5517426, 218.4589386
4: -131.3074951, 100.5786514, -131.1800690, 100.4843750, -231.7918701, 231.7587128
5: -116.4429474, 90.7623520, -116.3781738, 90.5326004, -206.9755554, 207.1405182
6: -112.0078583, 108.8155899, -111.9848404, 108.7961655, -220.8040161, 220.8004303
7: -122.7222748, 103.2880554, -122.6784973, 103.2570572, -225.9793243, 225.9665527
8: -149.1080322, 102.0858002, -149.0786896, 102.0288696, -251.1369019, 251.1644897
9: -111.9526443, 109.8554535, -111.8195267, 109.7964706, -221.7491150, 221.6749878

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1983285, upper bound: 206.1982635
time: 8.68 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1983285, upper bound: 206.1999718
time: 7.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 34.97 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 1, lower bound: -206.2038596, upper bound: 206.2066994
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 1, lower bound: -206.2038596, upper bound: 206.2066994
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 1, lower bound: -206.2049637, upper bound: 206.2079792
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 1, lower bound: -206.2049637, upper bound: 206.2085872
IS_B2_A2_B1, status: Status.VERIFIED, split count: 3, time: 34.97
Output dim: 1, lower bound: -206.1983285, upper bound: 206.1982635
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 1, lower bound: -206.1983285, upper bound: 206.1999718

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -122.3482819, 96.9952698, -120.8429642, 95.8095093, -218.1577911, 217.8382111
1: -104.3410645, 86.5778198, -103.0801620, 85.5215607, -189.8626251, 189.6579895
2: -135.7731781, 88.2661514, -134.1191406, 87.1916656, -222.9648285, 222.3852692
3: -143.5900574, 76.3189392, -141.8056488, 75.3910446, -218.9811096, 218.1245880
4: -132.0429077, 101.1205978, -130.4214172, 99.8890915, -231.9320068, 231.5420227
5: -117.0824127, 91.2991104, -115.6354446, 90.1722107, -207.2546234, 206.9345551
6: -112.6044006, 109.3904037, -111.2197113, 108.0560379, -220.6604309, 220.6101074
7: -123.3929977, 103.8461914, -121.8770294, 102.5779343, -225.9709320, 225.7232208
8: -149.9116974, 102.6137314, -148.0950775, 101.3835297, -251.2952271, 250.7087860
9: -112.5510635, 110.4543991, -111.1734009, 109.1022568, -221.6533051, 221.6278076

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1488634, upper bound: 206.1482282
time: 11.26 seconds

## Relational analysis of IS_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1441165, upper bound: 206.1437795
time: 10.06 seconds

## Relational analysis of IS_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2033995, upper bound: 206.2059658
time: 10.57 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2034227, upper bound: 206.2061140
time: 10.74 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -122.3482819, 96.9952698, -124.2648392, 98.5129242, -220.8612061, 221.2601013
1: -104.3410645, 86.5778198, -105.9577942, 87.9213104, -192.2623596, 192.5356140
2: -135.7731781, 88.2661514, -137.8892059, 89.6201706, -225.3933411, 226.1553192
3: -143.5900574, 76.3189392, -145.8034363, 77.5154266, -221.1054688, 222.1223755
4: -132.0429077, 101.1205978, -134.0974121, 102.6882553, -234.7311707, 235.2180176
5: -117.0824127, 91.2991104, -118.9212418, 92.6999207, -209.7823181, 210.2203522
6: -112.6044006, 109.3904037, -114.3840942, 111.1015015, -223.7059021, 223.7745056
7: -123.3929977, 103.8461914, -125.3163834, 105.4544601, -228.8474579, 229.1625519
8: -149.9116974, 102.6137314, -152.2157440, 104.1933365, -254.1050110, 254.8294525
9: -112.5510635, 110.4543991, -114.3122253, 112.1750336, -224.7261047, 224.7666321

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1488634, upper bound: 206.1486226
time: 11.60 seconds

## Relational analysis of IS_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1441165, upper bound: 206.1438236
time: 11.11 seconds

## Relational analysis of IS_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1864720, upper bound: 206.1889482
time: 9.99 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2038596, upper bound: 206.2066994
time: 9.55 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -125.7769241, 99.7046509, -120.8429642, 95.8095093, -221.5864258, 220.5475922
1: -107.2242966, 88.9826508, -103.0801620, 85.5215607, -192.7458496, 192.0628052
2: -139.5509796, 90.6998215, -134.1191406, 87.1916656, -226.7426147, 224.8189392
3: -147.5962524, 78.4478607, -141.8056488, 75.3910446, -222.9873047, 220.2535095
4: -135.7267151, 103.9251862, -130.4214172, 99.8890915, -235.6158142, 234.3466034
5: -120.3754959, 93.8318710, -115.6354446, 90.1722107, -210.5476990, 209.4673004
6: -115.7748718, 112.4425888, -111.2197113, 108.0560379, -223.8308868, 223.6622925
7: -126.8394623, 106.7288284, -121.8770294, 102.5779343, -229.4173889, 228.6058655
8: -154.0409698, 105.4287491, -148.0950775, 101.3835297, -255.4244995, 253.5238342
9: -115.6958160, 113.5333557, -111.1734009, 109.1022568, -224.7980194, 224.7067566

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1538833, upper bound: 206.1538315
time: 10.83 seconds

## Relational analysis of IS_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1413824, upper bound: 206.1419621
time: 10.34 seconds

## Relational analysis of IS_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1481933, upper bound: 206.1484009
time: 11.21 seconds

## Relational analysis of IS_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2045190, upper bound: 206.2073836
time: 9.74 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2044909, upper bound: 206.2073959
time: 11.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 105.27 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 105.27
Output dim: 1, lower bound: -206.2033995, upper bound: 206.2059658
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 105.27
Output dim: 1, lower bound: -206.2034227, upper bound: 206.2061140
IS_B1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 105.27
Output dim: 1, lower bound: -206.1864720, upper bound: 206.1889482
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 105.27
Output dim: 1, lower bound: -206.2038596, upper bound: 206.2066994
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 105.27
Output dim: 1, lower bound: -206.2045190, upper bound: 206.2073836
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 105.27
Output dim: 1, lower bound: -206.2044909, upper bound: 206.2073959
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 105.27
Output dim: 1, lower bound: -206.2049637, upper bound: 206.2085872
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 105.27
Output dim: 1, lower bound: -206.1983285, upper bound: 206.1999718
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=207.24490356445312
rel_dist={1: [-206.2534079160462, 206.25340791604617]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1947373, upper bound: 206.1930994
time: 11.02 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2528789, upper bound: 206.2528789
time: 9.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.78 seconds
IS_B1, status: Status.VERIFIED, split count: 1, time: 20.78
Output dim: 1, lower bound: -206.1947373, upper bound: 206.1930994
IS_B2, status: Status.UNKNOWN, split count: 1, time: 20.78
Output dim: 1, lower bound: -206.2528789, upper bound: 206.2528789

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -133.0105286, 105.4250412, -132.6288147, 105.1260223, -238.1365356, 238.0538483
1: -113.2106094, 94.0342941, -112.8885651, 93.7645645, -206.9751740, 206.9228516
2: -147.5505219, 95.8398438, -147.1331482, 95.5660858, -243.1166077, 242.9729919
3: -156.0251617, 82.8341141, -155.5809174, 82.5978394, -238.6230011, 238.4150238
4: -143.4313812, 109.8025894, -143.0204315, 109.4875412, -252.9189148, 252.8230286
5: -127.3003082, 99.2028809, -126.9403915, 98.9202576, -226.2205658, 226.1432343
6: -122.3499832, 118.8279724, -122.0004883, 118.4907303, -240.8407135, 240.8284607
7: -134.1060638, 112.8064041, -133.7233276, 112.4903870, -246.5964355, 246.5297089
8: -162.7028046, 111.3033218, -162.2402039, 110.9856262, -273.6883545, 273.5435181
9: -122.2542877, 119.9744720, -121.9091415, 119.6299515, -241.8842316, 241.8836060

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2036041, upper bound: 206.2024481
time: 12.54 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005862
time: 8.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.12 seconds
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 22.12
Output dim: 1, lower bound: -206.2036041, upper bound: 206.2024481
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 22.12
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005862

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -131.5043640, 104.2381363, -132.6288147, 105.1260223, -236.6303864, 236.8669434
1: -111.9491501, 92.9773102, -112.8885651, 93.7645645, -205.7136993, 205.8658600
2: -145.8954468, 94.7646484, -147.1331482, 95.5660858, -241.4615326, 241.8977661
3: -154.2395020, 81.9059525, -155.5809174, 82.5978394, -236.8373413, 237.4868774
4: -141.8089142, 108.5704422, -143.0204315, 109.4875412, -251.2964478, 251.5908813
5: -125.8522491, 98.0755768, -126.9403915, 98.9202576, -224.7725067, 225.0159607
6: -120.9647751, 117.4926453, -122.0004883, 118.4907303, -239.4555054, 239.4931335
7: -132.5888824, 111.5377655, -133.7233276, 112.4903870, -245.0792389, 245.2610931
8: -160.8848419, 110.0725784, -162.2402039, 110.9856262, -271.8704224, 272.3127747
9: -120.8762054, 118.6215515, -121.9091415, 119.6299515, -240.5061340, 240.5306854

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005864
time: 9.69 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005862
time: 10.01 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -124.2551422, 98.5684891, -122.4451523, 97.0896378, -221.3447571, 221.0135803
1: -105.9272766, 87.8787537, -104.3739090, 86.6119003, -192.5391693, 192.2526550
2: -137.9940186, 89.5577393, -135.9514160, 88.3093491, -226.3033752, 225.5091553
3: -145.5234070, 77.3668823, -143.5025330, 76.3190765, -221.8424835, 220.8694000
4: -133.9168549, 102.5689545, -132.0154877, 101.1512527, -235.0680847, 234.5844421
5: -118.8385925, 92.4406509, -117.1448212, 91.2778091, -210.1163635, 209.5854797
6: -114.3208389, 111.0633698, -112.6182785, 109.4569168, -223.7777557, 223.6816406
7: -125.2585983, 105.4193268, -123.4693146, 103.9251938, -229.1837769, 228.8885956
8: -152.1544647, 104.1155319, -149.9521179, 102.6588974, -254.8133545, 254.0676422
9: -114.1511459, 112.0845032, -112.5893784, 110.4694366, -224.6205750, 224.6738434

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 121

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005862
time: 9.56 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005864
time: 8.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.37 seconds
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 19.37
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005864
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 19.37
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005862
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 19.37
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005862
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 19.37
Output dim: 1, lower bound: -206.2005862, upper bound: 206.2005864

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -131.5043640, 104.2381363, -131.1220703, 103.9385910, -235.4429321, 235.3601990
1: -111.9491501, 92.9773102, -111.6265259, 92.7070770, -204.6562195, 204.6038055
2: -145.8954468, 94.7646484, -145.4772797, 94.4904175, -240.3858643, 240.2418976
3: -154.2395020, 81.9059525, -153.7945099, 81.6692810, -235.9087524, 235.7004700
4: -141.8089142, 108.5704422, -141.3972778, 108.2548218, -250.0637360, 249.9677124
5: -125.8522491, 98.0755768, -125.4917297, 97.7924728, -223.6447144, 223.5673065
6: -120.9647751, 117.4926453, -120.6147308, 117.1548386, -238.1196136, 238.1073456
7: -132.5888824, 111.5377655, -132.2054749, 111.2211838, -243.8100433, 243.7432404
8: -160.8848419, 110.0725784, -160.4214630, 109.7543564, -270.6391907, 270.4940186
9: -120.8762054, 118.6215515, -120.5304718, 118.2764435, -239.1526337, 239.1520081

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1955322, upper bound: 206.1944395
time: 11.55 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1959263, upper bound: 206.1947987
time: 10.75 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -131.5043640, 104.2381363, -123.8979950, 98.2882843, -229.7926331, 228.1361389
1: -111.9491501, 92.9773102, -105.6254959, 87.6261749, -199.5753021, 198.6027527
2: -145.8954468, 94.7646484, -137.6032104, 89.3014069, -235.1968536, 232.3678284
3: -154.2395020, 81.9059525, -145.1076202, 77.1451950, -231.3846741, 227.0135803
4: -141.8089142, 108.5704422, -133.5319672, 102.2740707, -244.0829773, 242.1024170
5: -125.8522491, 98.0755768, -118.5011673, 92.1760101, -218.0282593, 216.5767517
6: -120.9647751, 117.4926453, -113.9934158, 110.7479630, -231.7127380, 231.4860535
7: -132.5888824, 111.5377655, -124.9006958, 105.1234436, -237.7123260, 236.4384613
8: -160.8848419, 110.0725784, -151.7212830, 103.8177109, -264.7025146, 261.7938538
9: -120.8762054, 118.6215515, -113.8281403, 111.7617493, -232.6379242, 232.4496765

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1955322, upper bound: 206.1944393
time: 11.64 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1959263, upper bound: 206.1947987
time: 12.13 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -124.2551422, 98.5684891, -131.1076050, 103.9269409, -228.1820679, 229.6760712
1: -105.9272766, 87.8787537, -111.6150284, 92.6970444, -198.6243134, 199.4937744
2: -137.9940186, 89.5577393, -145.4616699, 94.4801941, -232.4742126, 235.0194092
3: -145.5234070, 77.3668823, -153.7779541, 81.6608353, -227.1842346, 231.1448212
4: -133.9168549, 102.5689545, -141.3822174, 108.2432098, -242.1600647, 243.9511719
5: -118.8385925, 92.4406509, -125.4780807, 97.7817459, -216.6203308, 217.9186859
6: -114.3208389, 111.0633698, -120.6016617, 117.1424332, -231.4632721, 231.6650391
7: -125.2585983, 105.4193268, -132.1906586, 111.2093658, -236.4679565, 237.6099548
8: -152.1544647, 104.1155319, -160.4040833, 109.7426834, -261.8970947, 264.5196228
9: -114.1511459, 112.0845032, -120.5175629, 118.2637863, -232.4149323, 232.6020508

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1926094, upper bound: 206.1925805
time: 9.99 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1929202, upper bound: 206.1929202
time: 8.63 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -124.2551422, 98.5684891, -123.8979950, 98.2882843, -222.5434113, 222.4664764
1: -105.9272766, 87.8787537, -105.6254959, 87.6261749, -193.5534515, 193.5042267
2: -137.9940186, 89.5577393, -137.6032104, 89.3014069, -227.2954254, 227.1609497
3: -145.5234070, 77.3668823, -145.1076202, 77.1451950, -222.6686096, 222.4744720
4: -133.9168549, 102.5689545, -133.5319672, 102.2740707, -236.1909180, 236.1009216
5: -118.8385925, 92.4406509, -118.5011673, 92.1760101, -211.0145874, 210.9418182
6: -114.3208389, 111.0633698, -113.9934158, 110.7479630, -225.0688019, 225.0567932
7: -125.2585983, 105.4193268, -124.9006958, 105.1234436, -230.3820496, 230.3200073
8: -152.1544647, 104.1155319, -151.7212830, 103.8177109, -255.9721527, 255.8367767
9: -114.1511459, 112.0845032, -113.8281403, 111.7617493, -225.9128571, 225.9126129

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1925805, upper bound: 206.1926094
time: 9.77 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1929202, upper bound: 206.1929202
time: 9.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.16 seconds
IS_B2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1955322, upper bound: 206.1944395
IS_B2_A1_B1_A2, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1959263, upper bound: 206.1947987
IS_B2_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1955322, upper bound: 206.1944393
IS_B2_A1_B2_A2, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1959263, upper bound: 206.1947987
IS_B2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1926094, upper bound: 206.1925805
IS_B2_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1929202, upper bound: 206.1929202
IS_B2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1925805, upper bound: 206.1926094
IS_B2_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 29.16
Output dim: 1, lower bound: -206.1929202, upper bound: 206.1929202
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=207.24490356445312
rel_dist={1: [-206.25298865227455, 206.2529886525199]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2044768, upper bound: 206.2067314
time: 11.38 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2008040, upper bound: 206.2008040
time: 8.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.84 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 19.84
Output dim: 1, lower bound: -206.2044768, upper bound: 206.2067314
IS_B2, status: Status.UNKNOWN, split count: 1, time: 19.84
Output dim: 1, lower bound: -206.2008040, upper bound: 206.2008040

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -133.0105286, 105.4250412, -131.5043640, 104.2381363, -237.2486420, 236.9293976
1: -113.2106094, 94.0342941, -111.9491501, 92.9773102, -206.1878967, 205.9834442
2: -147.5505219, 95.8398438, -145.8954468, 94.7646484, -242.3151398, 241.7352905
3: -156.0251617, 82.8341141, -154.2395020, 81.9059525, -237.9311218, 237.0735931
4: -143.4313812, 109.8025894, -141.8089142, 108.5704422, -252.0018311, 251.6115112
5: -127.3003082, 99.2028809, -125.8522491, 98.0755768, -225.3758850, 225.0551147
6: -122.3499832, 118.8279724, -120.9647751, 117.4926453, -239.8426208, 239.7927551
7: -134.1060638, 112.8064041, -132.5888824, 111.5377655, -245.6438293, 245.3952637
8: -162.7028046, 111.3033218, -160.8848419, 110.0725784, -272.7753296, 272.1881409
9: -122.2542877, 119.9744720, -120.8762054, 118.6215515, -240.8758087, 240.8506470

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2008040, upper bound: 206.2008040
time: 8.48 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2008040, upper bound: 206.2008040
time: 9.39 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -126.9109421, 100.6124420, -124.2551422, 98.5684891, -225.4794312, 224.8675842
1: -108.1109848, 89.7501373, -105.9272766, 87.8787537, -195.9897461, 195.6774139
2: -140.8534241, 91.4934235, -137.9940186, 89.5577393, -230.4111633, 229.4874420
3: -148.7913208, 79.0729065, -145.5234070, 77.3668823, -226.1581879, 224.5963135
4: -136.8404083, 104.8097534, -133.9168549, 102.5689545, -239.4093628, 238.7266083
5: -121.4337311, 94.6249771, -118.8385925, 92.4406509, -213.8743896, 213.4635468
6: -116.7311325, 113.4168472, -114.3208389, 111.0633698, -227.7944946, 227.7376862
7: -127.9637299, 107.6757431, -125.2585983, 105.4193268, -233.3830414, 232.9343414
8: -155.3431702, 106.3165970, -152.1544647, 104.1155319, -259.4586792, 258.4710388
9: -116.6720352, 114.4877472, -114.1511459, 112.0845032, -228.7565155, 228.6388397

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2008042, upper bound: 206.2008042
time: 8.03 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2008042, upper bound: 206.2008042
time: 7.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 40.57 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 1, lower bound: -206.2008040, upper bound: 206.2008040
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 1, lower bound: -206.2008040, upper bound: 206.2008040
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 1, lower bound: -206.2008042, upper bound: 206.2008042
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 40.57
Output dim: 1, lower bound: -206.2008042, upper bound: 206.2008042

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -131.5043640, 104.2381363, -131.5043640, 104.2381363, -235.7424927, 235.7424927
1: -111.9491501, 92.9773102, -111.9491501, 92.9773102, -204.9264069, 204.9264069
2: -145.8954468, 94.7646484, -145.8954468, 94.7646484, -240.6600647, 240.6600647
3: -154.2395020, 81.9059525, -154.2395020, 81.9059525, -236.1454468, 236.1454468
4: -141.8089142, 108.5704422, -141.8089142, 108.5704422, -250.3793488, 250.3793488
5: -125.8522491, 98.0755768, -125.8522491, 98.0755768, -223.9278259, 223.9278259
6: -120.9647751, 117.4926453, -120.9647751, 117.4926453, -238.4574280, 238.4574280
7: -132.5888824, 111.5377655, -132.5888824, 111.5377655, -244.1266479, 244.1266479
8: -160.8848419, 110.0725784, -160.8848419, 110.0725784, -270.9573975, 270.9573975
9: -120.8762054, 118.6215515, -120.8762054, 118.6215515, -239.4977264, 239.4977264

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1433469, upper bound: 206.1429410
time: 11.06 seconds

## Relational analysis of IS_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1964790, upper bound: 206.1986837
time: 10.30 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1968816, upper bound: 206.1990739
time: 10.98 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -124.2551422, 98.5684891, -131.5043640, 104.2381363, -228.4932556, 230.0728302
1: -105.9272766, 87.8787537, -111.9491501, 92.9773102, -198.9045563, 199.8278809
2: -137.9940186, 89.5577393, -145.8954468, 94.7646484, -232.7586365, 235.4531860
3: -145.5234070, 77.3668823, -154.2395020, 81.9059525, -227.4293518, 231.6063538
4: -133.9168549, 102.5689545, -141.8089142, 108.5704422, -242.4872894, 244.3778687
5: -118.8385925, 92.4406509, -125.8522491, 98.0755768, -216.9141541, 218.2929077
6: -114.3208389, 111.0633698, -120.9647751, 117.4926453, -231.8134766, 232.0281372
7: -125.2585983, 105.4193268, -132.5888824, 111.5377655, -236.7963562, 238.0081635
8: -152.1544647, 104.1155319, -160.8848419, 110.0725784, -262.2270203, 265.0003357
9: -114.1511459, 112.0845032, -120.8762054, 118.6215515, -232.7726746, 232.9606781

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2028162, upper bound: 206.2048017
time: 10.36 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2037652, upper bound: 206.2057853
time: 8.62 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -131.5043640, 104.2381363, -124.2551422, 98.5684891, -230.0728302, 228.4932556
1: -111.9491501, 92.9773102, -105.9272766, 87.8787537, -199.8278809, 198.9045563
2: -145.8954468, 94.7646484, -137.9940186, 89.5577393, -235.4531860, 232.7586365
3: -154.2395020, 81.9059525, -145.5234070, 77.3668823, -231.6063538, 227.4293518
4: -141.8089142, 108.5704422, -133.9168549, 102.5689545, -244.3778687, 242.4872894
5: -125.8522491, 98.0755768, -118.8385925, 92.4406509, -218.2929077, 216.9141541
6: -120.9647751, 117.4926453, -114.3208389, 111.0633698, -232.0281372, 231.8134766
7: -132.5888824, 111.5377655, -125.2585983, 105.4193268, -238.0081635, 236.7963562
8: -160.8848419, 110.0725784, -152.1544647, 104.1155319, -265.0003357, 262.2270203
9: -120.8762054, 118.6215515, -114.1511459, 112.0845032, -232.9606781, 232.7726746

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1980336, upper bound: 206.1981043
time: 8.17 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1999328, upper bound: 206.1999328
time: 8.42 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -124.2551422, 98.5684891, -124.2551422, 98.5684891, -222.8236084, 222.8236084
1: -105.9272766, 87.8787537, -105.9272766, 87.8787537, -193.8060150, 193.8060150
2: -137.9940186, 89.5577393, -137.9940186, 89.5577393, -227.5517578, 227.5517578
3: -145.5234070, 77.3668823, -145.5234070, 77.3668823, -222.8902893, 222.8902893
4: -133.9168549, 102.5689545, -133.9168549, 102.5689545, -236.4858093, 236.4858093
5: -118.8385925, 92.4406509, -118.8385925, 92.4406509, -211.2792053, 211.2792053
6: -114.3208389, 111.0633698, -114.3208389, 111.0633698, -225.3842163, 225.3842163
7: -125.2585983, 105.4193268, -125.2585983, 105.4193268, -230.6779022, 230.6779022
8: -152.1544647, 104.1155319, -152.1544647, 104.1155319, -256.2699585, 256.2699585
9: -114.1511459, 112.0845032, -114.1511459, 112.0845032, -226.2356110, 226.2356110

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 214

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 214

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1980336, upper bound: 206.1981043
time: 8.55 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1999328, upper bound: 206.1999328
time: 9.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 83.94 seconds
IS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.1964790, upper bound: 206.1986837
IS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.1968816, upper bound: 206.1990739
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.2028162, upper bound: 206.2048017
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.2037652, upper bound: 206.2057853
IS_B2_A1_A1, status: Status.VERIFIED, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.1980336, upper bound: 206.1981043
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.1999328, upper bound: 206.1999328
IS_B2_A2_A1, status: Status.VERIFIED, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.1980336, upper bound: 206.1981043
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 83.94
Output dim: 1, lower bound: -206.1999328, upper bound: 206.1999328

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -119.6373672, 94.9142990, -120.8429642, 95.8095093, -215.4468689, 215.7572632
1: -102.0954056, 84.6474533, -103.0801620, 85.5215607, -187.6169739, 187.7276154
2: -132.8789825, 86.2713547, -134.1191406, 87.1916656, -220.0706482, 220.3904877
3: -140.1401062, 74.5438919, -141.8056488, 75.3910446, -215.5311584, 216.3495178
4: -128.9920197, 98.8152466, -130.4214172, 99.8890915, -228.8811035, 229.2366638
5: -114.4083405, 89.0116196, -115.6354446, 90.1722107, -204.5805511, 204.6470490
6: -110.1097946, 106.9772339, -111.2197113, 108.0560379, -218.1658325, 218.1969452
7: -120.6138000, 101.5277252, -121.8770294, 102.5779343, -223.1917419, 223.4047394
8: -146.6116333, 100.3611679, -148.0950775, 101.3835297, -247.9951630, 248.4562378
9: -109.9508743, 107.9678345, -111.1734009, 109.1022568, -219.0531158, 219.1412354

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2019022, upper bound: 206.2038131
time: 8.61 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2019022, upper bound: 206.2048014
time: 10.62 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -120.5067215, 95.6022720, -124.2648392, 98.5129242, -219.0196533, 219.8670959
1: -102.8240356, 85.2591629, -105.9577942, 87.9213104, -190.7453156, 191.2169495
2: -133.8437195, 86.8925095, -137.8892059, 89.6201706, -223.4638977, 224.7816772
3: -141.1569214, 75.0873108, -145.8034363, 77.5154266, -218.6723328, 220.8907318
4: -129.9215240, 99.5258102, -134.0974121, 102.6882553, -232.6097717, 233.6232300
5: -115.2465439, 89.6551361, -118.9212418, 92.6999207, -207.9464417, 208.5763550
6: -110.9104462, 107.7539520, -114.3840942, 111.1015015, -222.0119476, 222.1380463
7: -121.4922180, 102.2629395, -125.3163834, 105.4544601, -226.9466858, 227.5792999
8: -147.6641235, 101.0690765, -152.2157440, 104.1933365, -251.8574371, 253.2848206
9: -110.7476578, 108.7441254, -114.3122253, 112.1750336, -222.9226685, 223.0563354

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2019112, upper bound: 206.2038131
time: 10.15 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2019112, upper bound: 206.2057856
time: 11.71 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -124.2648392, 98.5129242, -120.5067215, 95.6022720, -219.8670959, 219.0196533
1: -105.9577942, 87.9213104, -102.8240356, 85.2591629, -191.2169495, 190.7453156
2: -137.8892059, 89.6201706, -133.8437195, 86.8925095, -224.7816772, 223.4638977
3: -145.8034363, 77.5154266, -141.1569214, 75.0873108, -220.8907471, 218.6723328
4: -134.0974121, 102.6882553, -129.9215240, 99.5258102, -233.6232300, 232.6097717
5: -118.9212418, 92.6999207, -115.2465439, 89.6551361, -208.5763550, 207.9464417
6: -114.3840942, 111.1015015, -110.9104462, 107.7539520, -222.1380463, 222.0119476
7: -125.3163834, 105.4544601, -121.4922180, 102.2629395, -227.5792999, 226.9466858
8: -152.2157440, 104.1933365, -147.6641235, 101.0690765, -253.2848206, 251.8574371
9: -114.3122253, 112.1750336, -110.7476578, 108.7441254, -223.0563354, 222.9226685

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2038131, upper bound: 206.2019112
time: 10.83 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.2038131, upper bound: 206.2037652
time: 11.75 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -117.2118988, 92.9958191, -120.5067215, 95.6022720, -212.8141479, 213.5025330
1: -100.0976944, 82.9572906, -102.8240356, 85.2591629, -185.3568420, 185.7812958
2: -130.1967773, 84.5490799, -133.8437195, 86.8925095, -217.0892639, 218.3927917
3: -137.3169708, 73.0850677, -141.1569214, 75.0873108, -212.4042816, 214.2419891
4: -126.4083786, 96.8526306, -129.9215240, 99.5258102, -225.9341888, 226.7741547
5: -112.0899353, 87.2058334, -115.2465439, 89.6551361, -201.7450714, 202.4523621
6: -107.9126434, 104.8467484, -110.9104462, 107.7539520, -215.6665802, 215.7572021
7: -118.1824188, 99.4898758, -121.4922180, 102.2629395, -220.4453583, 220.9820862
8: -143.7176819, 98.3908615, -147.6641235, 101.0690765, -244.7867432, 246.0549774
9: -107.7588348, 105.8072662, -110.7476578, 108.7441254, -216.5029602, 216.5549011

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 234
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 151
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 151
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 121
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1981043, upper bound: 206.1980336
time: 8.35 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1981043, upper bound: 206.1999328
time: 9.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.89 seconds
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.2019022, upper bound: 206.2038131
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.2019022, upper bound: 206.2048014
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.2019112, upper bound: 206.2038131
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.2019112, upper bound: 206.2057856
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.2038131, upper bound: 206.2019112
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.2038131, upper bound: 206.2037652
IS_B2_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.1981043, upper bound: 206.1980336
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.89
Output dim: 1, lower bound: -206.1981043, upper bound: 206.1999328

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -113.7843399, 90.2824402, -120.8429642, 95.8095093, -209.5938263, 211.1253967
1: -97.2143402, 80.5522156, -103.0801620, 85.5215607, -182.7358856, 183.6323853
2: -126.4136200, 82.1134872, -134.1191406, 87.1916656, -213.6052856, 216.2326355
3: -133.3147125, 70.9541168, -141.8056488, 75.3910446, -208.7057495, 212.7597656
4: -122.7308578, 94.0462799, -130.4214172, 99.8890915, -222.6199493, 224.4676819
5: -108.7970352, 84.6707077, -115.6354446, 90.1722107, -198.9692230, 200.3061066
6: -104.7399216, 101.7933044, -111.2197113, 108.0560379, -212.7959442, 213.0130157
7: -114.7381134, 96.6086044, -121.8770294, 102.5779343, -217.3160400, 218.4856262
8: -139.5878601, 95.5726471, -148.0950775, 101.3835297, -240.9713898, 243.6676941
9: -104.6184158, 102.7255859, -111.1734009, 109.1022568, -213.7206116, 213.8989868

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 234
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1976995, upper bound: 206.1999521
time: 10.12 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -206.1968803, upper bound: 206.1987877
time: 11.49 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -117.2118988, 92.9958191, -120.8429642, 95.8095093, -213.0213776, 213.8387604
1: -100.0976944, 82.9572906, -103.0801620, 85.5215607, -185.6192627, 186.0374451
2: -130.1967773, 84.5490799, -134.1191406, 87.1916656, -217.3884430, 218.6682129
3: -137.3169708, 73.0850677, -141.8056488, 75.3910446, -212.7080078, 214.8907166
4: -126.4083786, 96.8526306, -130.4214172, 99.8890915, -226.2974701, 227.2740479
5: -112.0899353, 87.2058334, -115.6354446, 90.1722107, -202.2621460, 202.8412628
6: -107.9126434, 104.8467484, -111.2197113, 108.0560379, -215.9686737, 216.0664673
7: -118.1824188, 99.4898758, -121.8770294, 102.5779343, -220.7603455, 221.3669128
8: -143.7176819, 98.3908615, -148.0950775, 101.3835297, -245.1012115, 246.4859314
9: -107.7588348, 105.8072662, -111.1734009, 109.1022568, -216.8610687, 216.9806671

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 234
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 234
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 151
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 59
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 151
type: A, layer: 1, pos: 130
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 59
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 130
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 121
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1976995, upper bound: 206.1999521
time: 10.37 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -206.1968803, upper bound: 206.1997380
time: 10.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 57.16 seconds
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 57.16
Output dim: 1, lower bound: -206.1976995, upper bound: 206.1999521
IS_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 57.16
Output dim: 1, lower bound: -206.1968803, upper bound: 206.1987877
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 57.16
Output dim: 1, lower bound: -206.1976995, upper bound: 206.1999521
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 57.16
Output dim: 1, lower bound: -206.1968803, upper bound: 206.1997380
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 57.16
Output dim: 1, lower bound: -206.2019112, upper bound: 206.2038131
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 57.16
Output dim: 1, lower bound: -206.2019112, upper bound: 206.2057856
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 57.16
Output dim: 1, lower bound: -206.2038131, upper bound: 206.2019112
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 57.16
Output dim: 1, lower bound: -206.2038131, upper bound: 206.2037652
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 57.16
Output dim: 1, lower bound: -206.1981043, upper bound: 206.1999328
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=207.24490356445312
rel_dist={1: [-206.25322719906518, 206.25322719906507]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2047.32 seconds
