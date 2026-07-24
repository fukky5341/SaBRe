## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 154.56034074419998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957)
1: (-70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644)
2: (-94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537)
3: (-99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775)
4: (-103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431)
5: (-81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278)
6: (-83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274)
7: (-88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160)
8: (-104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331)
9: (-84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.97 + 14.46 = 15.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7150558, upper bound: 154.7150558

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6437439, upper bound: 154.6364807
time: 13.46 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6263043, upper bound: 154.6263043
time: 8.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 21.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 21.68
Output dim: 4, lower bound: -154.6437439, upper bound: 154.6364807
NS_A2, status: Status.UNKNOWN, split count: 1, time: 21.68
Output dim: 4, lower bound: -154.6263043, upper bound: 154.6263043

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -87.6555023, 68.3768311, -87.9151230, 68.5816803, -156.2371826, 156.2919464
1: -70.1716461, 62.0861206, -70.3884354, 62.2707710, -132.4423981, 132.4745178
2: -94.1626129, 64.2354660, -94.4465866, 64.4209595, -158.5835724, 158.6820374
3: -99.4909592, 54.9533005, -99.7921600, 55.1154213, -154.6063690, 154.7454376
4: -103.0291138, 65.1484680, -103.3198853, 65.3530502, -168.3821716, 168.4683228
5: -80.8302307, 65.7905731, -81.0754700, 65.9836578, -146.8138885, 146.8660126
6: -82.8069382, 77.1252670, -83.0478287, 77.3565063, -160.1634369, 160.1730652
7: -88.0284348, 74.8297195, -88.2885132, 75.0491028, -163.0775452, 163.1182251
8: -104.3845444, 72.1155777, -104.6999664, 72.3314667, -176.7160034, 176.8155518
9: -84.0119858, 75.1741028, -84.2536926, 75.4037476, -159.4156952, 159.4277802

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6240481, upper bound: 154.6164661
time: 11.46 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6131668, upper bound: 154.6035830
time: 10.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -99.6381149, 77.7250824, -87.2474747, 68.0554276, -167.6935120, 164.9725647
1: -79.7306213, 70.5418396, -69.8324356, 61.7963600, -141.5269318, 140.3742676
2: -107.1476135, 72.8780365, -93.7172241, 63.9433632, -171.0909424, 166.5952454
3: -113.0338058, 62.2619514, -99.0188751, 54.6976433, -167.7314453, 161.2808075
4: -117.3104477, 73.7910080, -102.5702438, 64.8297882, -182.1402283, 176.3612061
5: -91.8860626, 74.7644119, -80.4462433, 65.4875259, -157.3735962, 155.2106323
6: -94.1732941, 87.7177582, -82.4279099, 76.7625809, -170.9358521, 170.1456604
7: -100.0357056, 85.0350189, -87.6198730, 74.4847336, -174.5204315, 172.6548920
8: -118.5067368, 81.7048721, -103.8892441, 71.7757339, -190.2824402, 185.5941162
9: -95.5267792, 85.3593445, -83.6312256, 74.8153763, -170.3421478, 168.9905701

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5952179, upper bound: 154.5934074
time: 11.15 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 6.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 20.59 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.59
Output dim: 4, lower bound: -154.6240481, upper bound: 154.6164661
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.59
Output dim: 4, lower bound: -154.6131668, upper bound: 154.6035830
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 20.59
Output dim: 4, lower bound: -154.5952179, upper bound: 154.5934074
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 20.59
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -86.5459824, 67.4974747, -76.1074295, 59.2191505, -145.7651062, 143.6048889
1: -69.2417603, 61.3037567, -60.4897194, 53.9429703, -123.1847229, 121.7934723
2: -92.9429932, 63.4472427, -81.4660339, 56.0410385, -148.9840240, 144.9132690
3: -98.2028122, 54.2726326, -86.0722733, 47.8770294, -146.0798340, 140.3449097
4: -101.8439407, 64.2415543, -90.7329102, 55.6714897, -157.5154266, 154.9744568
5: -79.7717438, 64.9664383, -69.8064270, 57.2018166, -136.9735565, 134.7728577
6: -81.7932663, 76.1464920, -72.2550507, 66.9468994, -148.7401581, 148.4015503
7: -86.9255753, 73.9018707, -76.5544891, 65.1790924, -152.1046753, 150.4563599
8: -103.0481949, 71.1951675, -90.4823608, 62.5359726, -165.5841675, 161.6775208
9: -82.9993973, 74.1773071, -73.4856491, 64.7898788, -147.7892761, 147.6629639

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035170
time: 12.66 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035830
time: 12.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -81.8982849, 63.8293152, -79.1272888, 61.5440216, -143.4422913, 142.9566040
1: -65.3494415, 58.0205574, -62.8410416, 56.0389481, -121.3883896, 120.8615875
2: -87.8435440, 60.1394463, -84.7055740, 58.2168427, -146.0603638, 144.8450165
3: -92.8126221, 51.4170647, -89.4422150, 49.7415771, -142.5541534, 140.8592834
4: -96.8236694, 60.4723816, -94.4524384, 57.7357559, -154.5594177, 154.9247894
5: -75.3573151, 61.5067978, -72.5466690, 59.4146614, -134.7719727, 134.0534668
6: -77.5230713, 72.0439835, -75.1119308, 69.6036377, -147.1267090, 147.1559143
7: -82.2912064, 70.0015182, -79.5699844, 67.7204742, -150.0116882, 149.5714722
8: -97.4539261, 67.3574600, -94.0417480, 65.0077591, -162.4616699, 161.3991699
9: -78.7286072, 70.0092621, -76.4039841, 67.2536469, -145.9822388, 146.4132385

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035170
time: 10.82 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035830
time: 14.38 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -88.1771774, 68.6310730, -86.1373672, 67.1755447, -155.3527222, 154.7684326
1: -70.1039734, 62.4541550, -68.9019547, 61.0135574, -131.1175232, 131.3561096
2: -94.5377045, 64.7451706, -92.4969025, 63.1547737, -157.6924744, 157.2420654
3: -99.7077942, 55.2300072, -97.7299118, 54.0166359, -153.7244110, 152.9599152
4: -105.1213379, 64.3628006, -101.3844070, 63.9223747, -169.0436707, 165.7472076
5: -80.9323502, 66.2396240, -79.3871307, 64.6629562, -145.5953064, 145.6267548
6: -83.6983719, 77.6105957, -81.4136353, 75.7833099, -159.4816895, 159.0242310
7: -88.6391068, 75.4566193, -86.5163651, 73.5563660, -162.1954651, 161.9729919
8: -104.6999588, 72.1882782, -102.5521469, 70.8548050, -175.5547485, 174.7403870
9: -85.0876770, 75.0355759, -82.6181717, 73.8180313, -158.9056854, 157.6537476

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 6.75 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 6.63 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -90.8221741, 70.6583023, -81.5065689, 63.5204010, -154.3425751, 152.1648712
1: -72.1524200, 64.2817154, -65.0234756, 57.7420731, -129.8945007, 129.3051910
2: -97.3637238, 66.6522293, -87.4158554, 59.8589745, -157.2227020, 154.0680847
3: -102.6524048, 56.8709373, -92.3590393, 51.1712036, -153.8235931, 149.2299805
4: -108.4133377, 66.1444931, -96.3837128, 60.1655121, -168.5788574, 162.5281982
5: -83.3201828, 68.1759415, -74.9886246, 61.2153854, -144.5355682, 143.1645050
6: -86.2082520, 79.9323273, -77.1587524, 71.6960068, -157.9041901, 157.0910797
7: -91.2798157, 77.6821976, -81.8988495, 69.6702805, -160.9500732, 159.5810547
8: -107.8128662, 74.3655243, -96.9778290, 67.0306854, -174.8435364, 171.3433533
9: -87.6521988, 77.1712036, -78.3631058, 69.6643219, -157.3165283, 155.5343018

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 8.23 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
time: 6.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.62 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035170
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.6129785, upper bound: 154.6035830
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035170
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.6129777, upper bound: 154.6035830
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 15.62
Output dim: 4, lower bound: -154.5912838, upper bound: 154.5912838

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -75.8630447, 59.0259247, -76.1074295, 59.2191505, -135.0821686, 135.1333618
1: -60.2853775, 53.7690048, -60.4897194, 53.9429703, -114.2283478, 114.2587128
2: -81.1988144, 55.8664894, -81.4660339, 56.0410385, -137.2398376, 137.3325195
3: -85.7888718, 47.7239532, -86.0722733, 47.8770294, -133.6658783, 133.7962341
4: -90.4597092, 55.4781036, -90.7329102, 55.6714897, -146.1311951, 146.2109833
5: -69.5755997, 57.0195541, -69.8064270, 57.2018166, -126.7774124, 126.8259811
6: -72.0279388, 66.7294312, -72.2550507, 66.9468994, -138.9748383, 138.9844818
7: -76.3097153, 64.9726639, -76.5544891, 65.1790924, -141.4888000, 141.5271606
8: -90.1852646, 62.3321877, -90.4823608, 62.5359726, -152.7212372, 152.8145447
9: -73.2582016, 64.5734177, -73.4856491, 64.7898788, -138.0480804, 138.0590668

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
time: 12.46 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
time: 11.14 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -78.8722000, 61.3423080, -76.1074295, 59.2191505, -138.0913239, 137.4497375
1: -62.6281967, 55.8573456, -60.4897194, 53.9429703, -116.5711670, 116.3470459
2: -84.4262772, 58.0346298, -81.4660339, 56.0410385, -140.4673157, 139.5006714
3: -89.1462402, 49.5819397, -86.0722733, 47.8770294, -137.0232544, 135.6542053
4: -94.1661987, 57.5342522, -90.7329102, 55.6714897, -149.8376617, 148.2671661
5: -72.3057098, 59.2242889, -69.8064270, 57.2018166, -129.5074921, 129.0307007
6: -74.8744659, 69.3764191, -72.2550507, 66.9468994, -141.8213501, 141.6314697
7: -79.3141174, 67.5049286, -76.5544891, 65.1790924, -144.4932098, 144.0594177
8: -93.7315140, 64.7954712, -90.4823608, 62.5359726, -156.2674866, 155.2778320
9: -76.1658707, 67.0278549, -73.4856491, 64.7898788, -140.9557495, 140.5135040

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
time: 13.63 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
time: 12.40 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -75.8630447, 59.0259247, -79.1272888, 61.5440216, -137.4070435, 138.1532135
1: -60.2853775, 53.7690048, -62.8410416, 56.0389481, -116.3243256, 116.6100464
2: -81.1988144, 55.8664894, -84.7055740, 58.2168427, -139.4156494, 140.5720673
3: -85.7888718, 47.7239532, -89.4422150, 49.7415771, -135.5304260, 137.1661682
4: -90.4597092, 55.4781036, -94.4524384, 57.7357559, -148.1954651, 149.9305115
5: -69.5755997, 57.0195541, -72.5466690, 59.4146614, -128.9902649, 129.5662231
6: -72.0279388, 66.7294312, -75.1119308, 69.6036377, -141.6315765, 141.8413544
7: -76.3097153, 64.9726639, -79.5699844, 67.7204742, -144.0301819, 144.5426331
8: -90.1852646, 62.3321877, -94.0417480, 65.0077591, -155.1930237, 156.3739319
9: -73.2582016, 64.5734177, -76.4039841, 67.2536469, -140.5118408, 140.9774017

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6009362, upper bound: 154.5936612
time: 12.60 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971538
time: 10.46 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -78.8722000, 61.3423080, -79.1272888, 61.5440216, -140.4161987, 140.4695892
1: -62.6281967, 55.8573456, -62.8410416, 56.0389481, -118.6671448, 118.6983643
2: -84.4262772, 58.0346298, -84.7055740, 58.2168427, -142.6431274, 142.7402039
3: -89.1462402, 49.5819397, -89.4422150, 49.7415771, -138.8878021, 139.0241547
4: -94.1661987, 57.5342522, -94.4524384, 57.7357559, -151.9019318, 151.9866791
5: -72.3057098, 59.2242889, -72.5466690, 59.4146614, -131.7203522, 131.7709656
6: -74.8744659, 69.3764191, -75.1119308, 69.6036377, -144.4781036, 144.4883423
7: -79.3141174, 67.5049286, -79.5699844, 67.7204742, -147.0345917, 147.0748749
8: -93.7315140, 64.7954712, -94.0417480, 65.0077591, -158.7392731, 158.8372040
9: -76.1658707, 67.0278549, -76.4039841, 67.2536469, -143.4195251, 143.4318390

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 185

### Candidate
type: A, layer: 1, pos: 185

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5985971, upper bound: 154.5893743
time: 11.02 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971874
time: 10.40 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -88.1771774, 68.6310730, -75.4747543, 58.7194176, -146.8965912, 144.1058044
1: -70.1039734, 62.4541550, -59.9623299, 53.4928551, -123.5968170, 122.4164734
2: -94.5377045, 64.7451706, -80.7750092, 55.5882759, -150.1259766, 145.5201721
3: -99.7077942, 55.2300072, -85.3398819, 47.4797287, -147.1875305, 140.5698853
4: -105.1213379, 64.3628006, -90.0229874, 55.1739922, -160.2953033, 154.3857880
5: -80.9323502, 66.2396240, -69.2102737, 56.7303314, -137.6626892, 135.4498901
6: -83.6983719, 77.6105957, -71.6662827, 66.3844757, -150.0828247, 149.2768555
7: -88.6391068, 75.4566193, -75.9207764, 64.6442795, -153.2833862, 151.3773956
8: -104.6999588, 72.1882782, -89.7133026, 62.0076180, -166.7075806, 161.9015808
9: -85.0876770, 75.0355759, -72.8953934, 64.2318954, -149.3195801, 147.9309692

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
time: 9.73 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
time: 9.40 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -88.1771774, 68.6310730, -78.4903641, 61.0406799, -149.2178650, 147.1214294
1: -70.1039734, 62.4541550, -62.3113670, 55.5858994, -125.6898346, 124.7655106
2: -94.5377045, 64.7451706, -84.0092316, 57.7605438, -152.2982330, 148.7543945
3: -99.7077942, 55.2300072, -88.7050476, 49.3415375, -149.0493317, 143.9350586
4: -105.1213379, 64.3628006, -93.7343063, 57.2366791, -162.3579712, 158.0971069
5: -80.9323502, 66.2396240, -71.9469604, 58.9394646, -139.8718109, 138.1865845
6: -83.6983719, 77.6105957, -74.5181122, 69.0370255, -152.7353973, 152.1286774
7: -88.6391068, 75.4566193, -78.9308853, 67.1816483, -155.8207550, 154.3875122
8: -104.6999588, 72.1882782, -93.2673187, 64.4765930, -169.1765442, 165.4555511
9: -85.0876770, 75.0355759, -75.8076401, 66.6926346, -151.7802734, 150.8432159

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
time: 8.55 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
time: 9.40 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -90.8221741, 70.6583023, -75.4747543, 58.7194176, -149.5415802, 146.1330261
1: -72.1524200, 64.2817154, -59.9623299, 53.4928551, -125.6452484, 124.2440414
2: -97.3637238, 66.6522293, -80.7750092, 55.5882759, -152.9519958, 147.4272156
3: -102.6524048, 56.8709373, -85.3398819, 47.4797287, -150.1321411, 142.2108154
4: -108.4133377, 66.1444931, -90.0229874, 55.1739922, -163.5873108, 156.1674805
5: -83.3201828, 68.1759415, -69.2102737, 56.7303314, -140.0505066, 137.3862152
6: -86.2082520, 79.9323273, -71.6662827, 66.3844757, -152.5926666, 151.5986023
7: -91.2798157, 77.6821976, -75.9207764, 64.6442795, -155.9240875, 153.6029663
8: -107.8128662, 74.3655243, -89.7133026, 62.0076180, -169.8204803, 164.0788269
9: -87.6521988, 77.1712036, -72.8953934, 64.2318954, -151.8840942, 150.0665588

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
time: 12.96 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
time: 8.72 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -90.8221741, 70.6583023, -78.4903641, 61.0406799, -151.8628387, 149.1486664
1: -72.1524200, 64.2817154, -62.3113670, 55.5858994, -127.7382812, 126.5930786
2: -97.3637238, 66.6522293, -84.0092316, 57.7605438, -155.1242676, 150.6614380
3: -102.6524048, 56.8709373, -88.7050476, 49.3415375, -151.9939270, 145.5759888
4: -108.4133377, 66.1444931, -93.7343063, 57.2366791, -165.6499786, 159.8787842
5: -83.3201828, 68.1759415, -71.9469604, 58.9394646, -142.2596436, 140.1228790
6: -86.2082520, 79.9323273, -74.5181122, 69.0370255, -155.2452393, 154.4504395
7: -91.2798157, 77.6821976, -78.9308853, 67.1816483, -158.4614563, 156.6130829
8: -107.8128662, 74.3655243, -93.2673187, 64.4765930, -172.2894287, 167.6327972
9: -87.6521988, 77.1712036, -75.8076401, 66.6926346, -154.3448334, 152.9788361

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
time: 7.65 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
time: 11.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 20.28 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.6153702, upper bound: 154.6082560
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.6173890, upper bound: 154.6098976
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.6009362, upper bound: 154.5936612
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971538
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5985971, upper bound: 154.5893743
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.6063430, upper bound: 154.5971874
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5842914, upper bound: 154.5832181
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5885594, upper bound: 154.5868058
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5770309, upper bound: 154.5779410
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 20.28
Output dim: 4, lower bound: -154.5846246, upper bound: 154.5846246

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -72.2446747, 56.1646576, -66.5325394, 51.6589584, -123.9036331, 122.6971970
1: -57.2325706, 51.2009277, -52.4435539, 47.1499557, -104.3825226, 103.6444855
2: -77.2411880, 53.3007317, -71.0065384, 49.2442207, -126.4854126, 124.3072662
3: -81.6033325, 45.4756927, -75.0224991, 41.9286957, -123.5320206, 120.4981918
4: -86.5312576, 52.5105324, -80.2726822, 47.8808823, -134.4121246, 132.7832184
5: -66.1475143, 54.3139877, -60.7530212, 50.0444679, -116.1919708, 115.0670090
6: -68.6661682, 63.5286484, -63.3492393, 58.4781990, -127.1443634, 126.8778839
7: -72.6856384, 61.9359207, -66.9698410, 57.1305695, -129.8161926, 128.9057617
8: -85.7855530, 59.2655754, -78.8642807, 54.4325676, -140.2181244, 138.1298523
9: -69.9307327, 61.3171158, -64.6489258, 56.2133293, -126.1440582, 125.9660187

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6217867, upper bound: 154.6139410
time: 16.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6217867, upper bound: 154.6146722
time: 12.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -73.7258072, 57.3388710, -70.7142105, 54.9598579, -128.6856689, 128.0530701
1: -58.4858818, 52.2545967, -55.9476471, 50.1179848, -108.6038513, 108.2022324
2: -78.8644104, 54.3523026, -75.5692978, 52.2213478, -131.0857544, 129.9216003
3: -83.3191833, 46.3961258, -79.8393097, 44.5260658, -127.8452454, 126.2354202
4: -88.1393585, 53.7304688, -84.8793640, 51.2546043, -139.3939209, 138.6098022
5: -67.5536880, 55.4225388, -64.7002869, 53.1705017, -120.7241898, 120.1228180
6: -70.0442047, 64.8406754, -67.2483826, 62.1759148, -132.2201233, 132.0890350
7: -74.1715927, 63.1791687, -71.1540756, 60.6520348, -134.8236237, 134.3332520
8: -87.5901718, 60.5267143, -83.9303055, 57.9791565, -145.5693359, 144.4570160
9: -71.2929306, 62.6540565, -68.5222244, 59.9433098, -131.2362366, 131.1762848

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6219062, upper bound: 154.6142783
time: 11.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6219062, upper bound: 154.6160564
time: 11.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -75.2286224, 58.4575043, -66.5325394, 51.6589584, -126.8875580, 124.9900436
1: -59.5535393, 53.2683601, -52.4435539, 47.1499557, -106.7034912, 105.7119141
2: -80.4365158, 55.4481277, -71.0065384, 49.2442207, -129.6807404, 126.4546509
3: -84.9344635, 47.3154411, -75.0224991, 41.9286957, -126.8631516, 122.3379288
4: -90.2021332, 54.5470734, -80.2726822, 47.8808823, -138.0830078, 134.8197327
5: -68.8517532, 56.4999275, -60.7530212, 50.0444679, -118.8962250, 117.2529449
6: -71.4858246, 66.1466217, -63.3492393, 58.4781990, -129.9640045, 129.4958496
7: -75.6588745, 64.4440613, -66.9698410, 57.1305695, -132.7894135, 131.4139099
8: -89.2929688, 61.7100792, -78.8642807, 54.4325676, -143.7255096, 140.5743561
9: -72.8075104, 63.7494812, -64.6489258, 56.2133293, -129.0208282, 128.3984070

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6135560, upper bound: 154.6071832
time: 17.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6135560, upper bound: 154.6082560
time: 12.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -76.7928772, 59.6998024, -70.7142105, 54.9598579, -131.7527313, 130.4140015
1: -60.8784485, 54.3818855, -55.9476471, 50.1179848, -110.9964294, 110.3295288
2: -82.1516342, 56.5608215, -75.5692978, 52.2213478, -134.3729858, 132.1301270
3: -86.7446289, 48.2894974, -79.8393097, 44.5260658, -131.2706909, 128.1288147
4: -91.9015732, 55.8351898, -84.8793640, 51.2546043, -143.1561432, 140.7145386
5: -70.3378143, 57.6708488, -64.7002869, 53.1705017, -123.5083160, 122.3711319
6: -72.9420013, 67.5352631, -67.2483826, 62.1759148, -135.1179199, 134.7836456
7: -77.2300339, 65.7598343, -71.1540756, 60.6520348, -137.8820648, 136.9139099
8: -91.2024307, 63.0393600, -83.9303055, 57.9791565, -149.1815796, 146.9696655
9: -74.2485886, 65.1613770, -68.5222244, 59.9433098, -134.1918793, 133.6835938

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6136257, upper bound: 154.6074682
time: 28.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6136257, upper bound: 154.6098976
time: 10.02 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -66.2910919, 51.4681358, -75.4831696, 58.6588860, -124.9499588, 126.9512711
1: -52.2418594, 46.9781151, -59.7659492, 53.4495926, -105.6914444, 106.7440643
2: -70.7424850, 49.0714874, -80.7150345, 55.6300697, -126.3725281, 129.7865143
3: -74.7429428, 41.7774658, -85.2295532, 47.4747543, -122.2176819, 127.0070190
4: -80.0030212, 47.6896477, -90.4875793, 54.7480545, -134.7510681, 138.1772308
5: -60.5249977, 49.8640785, -69.0923004, 56.6899414, -117.2149353, 118.9563751
6: -63.1249008, 58.2633362, -71.7227249, 66.3734055, -129.4983063, 129.9860382
7: -66.7279816, 56.9266472, -75.9140930, 64.6590652, -131.3870544, 132.8407440
8: -78.5711823, 54.2308617, -89.6022797, 61.9219513, -140.4931335, 143.8331451
9: -64.4237823, 55.9998665, -73.0451202, 63.9746361, -128.3983917, 129.0449829

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5979257, upper bound: 154.5888262
time: 15.54 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5979257, upper bound: 154.5942628
time: 11.27 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -70.4696350, 54.7664070, -77.0483551, 59.9018250, -130.3714600, 131.8147583
1: -55.7431908, 49.9438286, -61.0915680, 54.5636940, -110.3068848, 111.0354004
2: -75.3017960, 52.0465317, -82.4312592, 56.7432938, -132.0450897, 134.4777832
3: -79.5557709, 44.3727722, -87.0408554, 48.4494133, -128.0051880, 131.4136200
4: -84.6058884, 51.0610161, -92.1881180, 56.0370560, -140.6429443, 143.2491302
5: -64.4691391, 52.9878540, -70.5791855, 57.8615189, -122.3306580, 123.5670319
6: -67.0210190, 61.9581642, -73.1797180, 67.7628632, -134.7838745, 135.1378784
7: -70.9090195, 60.4454651, -77.4862595, 65.9756546, -136.8846741, 137.9317169
8: -83.6330414, 57.7749138, -91.5132294, 63.2518883, -146.8849335, 149.2881470
9: -68.2942352, 59.7267723, -74.4870148, 65.3874359, -133.6816711, 134.2137756

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5990378, upper bound: 154.5896164
time: 15.34 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5990378, upper bound: 154.5973312
time: 17.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -75.2286224, 58.4575043, -69.3515167, 53.8198586, -129.0484772, 127.8090134
1: -59.5535393, 53.2683601, -54.6287613, 49.1025925, -108.6561279, 107.8971252
2: -80.4365158, 55.4481277, -74.0247421, 51.2685165, -131.7050171, 129.4728394
3: -84.9344635, 47.3154411, -78.1715851, 43.6644440, -128.5989075, 125.4870224
4: -90.2021332, 54.5470734, -83.7478714, 49.7958145, -139.9979401, 138.2949219
5: -68.8517532, 56.4999275, -63.3053360, 52.1042290, -120.9559784, 119.8052673
6: -71.4858246, 66.1466217, -66.0128937, 60.9487495, -132.4345703, 132.1595001
7: -75.6588745, 64.4440613, -69.7757797, 59.4958878, -135.1547546, 134.2198334
8: -89.2929688, 61.7100792, -82.1745377, 56.7382202, -146.0311737, 143.8845825
9: -72.8075104, 63.7494812, -67.3637848, 58.5042763, -131.3117676, 131.1132660

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5971611, upper bound: 154.5884014
time: 9.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.5971611, upper bound: 154.5893743
time: 11.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -76.7928772, 59.6998024, -73.8993988, 57.4132385, -134.2061157, 133.5991669
1: -60.8784485, 54.3818855, -58.4407692, 52.3292694, -113.2077179, 112.8226547
2: -82.1516342, 56.5608215, -78.9877777, 54.5110092, -136.6626434, 135.5485992
3: -86.7446289, 48.2894974, -83.4078979, 46.4913139, -133.2359467, 131.6973877
4: -91.9015732, 55.8351898, -88.7636185, 53.4638672, -145.3654175, 144.5988159
5: -70.3378143, 57.6708488, -67.5984268, 55.5078506, -125.8456650, 125.2692719
6: -72.9420013, 67.5352631, -70.2550659, 64.9728699, -137.9148712, 137.7903137
7: -77.2300339, 65.7598343, -74.3310394, 63.3316879, -140.5617218, 140.0908661
8: -91.2024307, 63.0393600, -87.6850967, 60.5926666, -151.7950897, 150.7244415
9: -74.2485886, 65.1613770, -71.5839920, 62.5602379, -136.8088074, 136.7453613

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6009362, upper bound: 154.5936612
time: 11.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6009362, upper bound: 154.5971875
time: 12.82 seconds

## BFS NS instance: NS_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -78.7053528, 61.1492691, -71.8567886, 55.8583870, -134.5637360, 133.0060577
1: -62.1427269, 55.7332268, -56.9098282, 50.9249001, -113.0676193, 112.6430435
2: -84.1807098, 58.0214195, -76.8178482, 53.0225143, -137.2031860, 134.8392639
3: -88.7659378, 49.3476944, -81.1547623, 45.2315750, -133.9974976, 130.5024567
4: -94.7875366, 56.6400757, -86.0950241, 52.2068672, -146.9943848, 142.7351074
5: -71.9717331, 59.1683235, -65.7824173, 54.0247650, -125.9964981, 124.9507446
6: -74.8825302, 69.2337341, -68.3047180, 63.1840248, -138.0665436, 137.5384369
7: -79.1562119, 67.4981384, -72.2970200, 61.6078682, -140.7640686, 139.7951660
8: -93.1994629, 64.1741257, -85.3144760, 58.9408951, -152.1403503, 149.4886017
9: -76.3576813, 66.5386200, -69.5678940, 60.9759102, -137.3335724, 136.1065063

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 171
type: A, layer: 1, pos: 171
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 212
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6036240, upper bound: 154.6036240
time: 7.06 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6036240, upper bound: 154.6039603
time: 7.22 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 15.43 + 587.71 = 603.14 seconds
