## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 197.2433907684


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486)
1: (-87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961)
2: (-114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986)
3: (-122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255)
4: (-112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691)
5: (-100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425)
6: (-96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658)
7: (-105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948)
8: (-125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392)
9: (-96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.90 + 8.12 = 9.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4408316, upper bound: 197.4408316

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3406430, upper bound: 197.3402623
time: 7.67 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3232398, upper bound: 197.3232398
time: 5.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.85
Output dim: 4, lower bound: -197.3406430, upper bound: 197.3402623
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.85
Output dim: 4, lower bound: -197.3232398, upper bound: 197.3232398

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -104.3831787, 82.8770676, -104.5059052, 82.9751511, -187.3583069, 187.3829651
1: -87.0132828, 73.6159668, -87.1186676, 73.7036362, -160.7169189, 160.7346344
2: -114.7991562, 74.9771271, -114.9358521, 75.0652771, -189.8644257, 189.9129333
3: -122.2485809, 64.2338867, -122.3928833, 64.3120346, -186.5606079, 186.6267548
4: -112.3497925, 86.3700867, -112.4815369, 86.4718399, -198.8216248, 198.8516235
5: -100.2480621, 78.1116562, -100.3661041, 78.2024307, -178.4505005, 178.4777527
6: -96.5637970, 92.2520523, -96.6764297, 92.3622818, -188.9260559, 188.9284515
7: -105.2925644, 88.3614044, -105.4159775, 88.4643326, -193.7568512, 193.7773743
8: -125.8182297, 86.0569229, -125.9699860, 86.1614532, -211.9796753, 212.0269012
9: -96.0615463, 94.3137054, -96.1728745, 94.4252930, -190.4868317, 190.4865723

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2965964, upper bound: 197.2958543
time: 7.84 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3053524, upper bound: 197.3024921
time: 7.14 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3381278, upper bound: 197.3373447
time: 8.19 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -111.3679199, 88.4211121, -104.1239395, 82.6704330, -194.0383453, 192.5450439
1: -92.6760406, 78.4514008, -86.7910843, 73.4313736, -166.1074219, 165.2424469
2: -122.4418869, 79.8888092, -114.5107193, 74.7911148, -197.2329712, 194.3995361
3: -130.4202118, 68.2964859, -121.9440689, 64.0690765, -194.4892731, 190.2405548
4: -119.9568024, 91.9793015, -112.0713959, 86.1551437, -206.1119385, 204.0506897
5: -107.0011215, 83.3186722, -99.9985733, 77.9203491, -184.9214783, 183.3172150
6: -103.0942459, 98.3950500, -96.3256912, 92.0198364, -195.1140747, 194.7206879
7: -112.2794952, 94.2552719, -105.0316467, 88.1441269, -200.4236145, 199.2869110
8: -134.0392761, 91.4946136, -125.4985504, 85.8366852, -219.8759613, 216.9931641
9: -102.4540939, 100.5486450, -95.8263245, 94.0785751, -196.5326691, 196.3749542

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2849843, upper bound: 197.2859300
time: 7.10 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3205095, upper bound: 197.3205095
time: 4.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.32 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.32
Output dim: 4, lower bound: -197.3053524, upper bound: 197.3024921
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.32
Output dim: 4, lower bound: -197.3381278, upper bound: 197.3373447
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 36.32
Output dim: 4, lower bound: -197.2849843, upper bound: 197.2859300
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 36.32
Output dim: 4, lower bound: -197.3205095, upper bound: 197.3205095

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -99.1582718, 78.7655029, -89.6735382, 71.3006516, -170.4589233, 168.4390106
1: -82.6743851, 69.9671783, -74.8050461, 63.3510971, -146.0254517, 144.7722168
2: -109.0606232, 71.2860947, -98.6563034, 64.5893326, -173.6499634, 169.9423828
3: -116.1441193, 61.0260849, -105.0731049, 55.2181587, -171.3622742, 166.0991821
4: -106.7588501, 82.1233444, -96.6044388, 74.4187164, -181.1775665, 178.7277679
5: -95.2567673, 74.2553940, -86.2035370, 67.2643814, -162.5211182, 160.4589081
6: -91.8084412, 87.6475525, -83.1557770, 79.2965851, -171.1050262, 170.8033295
7: -100.0595398, 84.0290909, -90.5638504, 76.1672058, -176.2267456, 174.5929413
8: -119.5546875, 81.7636185, -108.1830673, 73.9889526, -193.5436401, 189.9466858
9: -91.2963867, 89.5949173, -82.6571503, 81.0513077, -172.3476868, 172.2520599

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3046504, upper bound: 197.3020683
time: 6.93 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3046504, upper bound: 197.3024921
time: 7.41 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -101.6034164, 80.6898880, -97.3372955, 77.3335419, -178.9369507, 178.0271912
1: -84.6987839, 71.6745453, -81.1477661, 68.6961136, -153.3948669, 152.8222961
2: -111.7461472, 73.0156097, -107.0625916, 70.0063400, -181.7524872, 180.0781860
3: -119.0052719, 62.5306816, -114.0295410, 59.9184914, -178.9237213, 176.5602264
4: -109.3746109, 84.1041794, -104.8082352, 80.6281815, -190.0027924, 188.9124146
5: -97.5910797, 76.0601273, -93.5135345, 72.9115753, -170.5026398, 169.5736694
6: -94.0319824, 89.8001709, -90.1470566, 86.0385056, -180.0704956, 179.9472198
7: -102.5143509, 86.0543671, -98.2523117, 82.5152283, -185.0295715, 184.3066711
8: -122.4816360, 83.7704926, -117.3670349, 80.2623596, -202.7439880, 201.1375275
9: -93.5253220, 91.8039093, -89.6321716, 87.9542084, -181.4795227, 181.4360809

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3290026, upper bound: 197.3282470
time: 7.81 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3290026, upper bound: 197.3373447
time: 8.26 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -96.6377411, 76.8255386, -98.8988113, 78.5587006, -175.1964264, 175.7243347
1: -80.4424744, 68.1649094, -82.4519348, 69.7824020, -150.2248535, 150.6168518
2: -106.2736511, 69.4814224, -108.7720261, 71.0999146, -177.3735657, 178.2534485
3: -113.2210770, 59.2589722, -115.8393707, 60.8610420, -174.0821228, 175.0983429
4: -104.1916580, 80.0082245, -106.4803467, 81.9082642, -186.0999146, 186.4885559
5: -92.9357910, 72.4521179, -95.0071716, 74.0639572, -166.9997559, 167.4592896
6: -89.6652679, 85.4194107, -91.5701904, 87.4152374, -177.0804749, 176.9895935
7: -97.5261002, 82.0366669, -99.7983856, 83.8116837, -181.3377838, 181.8350525
8: -116.3683319, 79.4070282, -119.2348328, 81.5431595, -197.9114990, 198.6418457
9: -89.0305252, 87.2649460, -91.0609894, 89.3596649, -178.3901825, 178.3259277

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2786441
time: 4.42 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2859300
time: 4.57 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -104.2336349, 82.8083878, -101.3476028, 80.4860458, -184.7196655, 184.1559906
1: -86.7353592, 73.4676285, -84.4795456, 71.4923859, -158.2277527, 157.9471588
2: -114.6076736, 74.8536148, -111.4615860, 72.8320694, -187.4397430, 186.3152008
3: -122.0956421, 63.9222336, -118.7047424, 62.3679848, -184.4636230, 182.6269836
4: -112.3205109, 86.1664200, -109.0999832, 83.8921356, -196.2126465, 195.2664032
5: -100.1814423, 78.0530777, -97.3449707, 75.8714066, -176.0528564, 175.3980408
6: -96.5953064, 92.1027374, -93.7970276, 89.5710907, -186.1663971, 185.8997498
7: -105.1498260, 88.3351059, -102.2568817, 85.8400116, -190.9898376, 190.5919495
8: -125.4759293, 85.6256790, -122.1662140, 83.5531158, -209.0290375, 207.7918701
9: -95.9430618, 94.1088257, -93.2931976, 91.5719757, -187.5149994, 187.4020233

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2859300, upper bound: 197.2849843
time: 6.67 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2859300, upper bound: 197.3205095
time: 6.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.61 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.3046504, upper bound: 197.3020683
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.3046504, upper bound: 197.3024921
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.3290026, upper bound: 197.3282470
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.3290026, upper bound: 197.3373447
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2786441
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2859300
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.2859300, upper bound: 197.2849843
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 14.61
Output dim: 4, lower bound: -197.2859300, upper bound: 197.3205095

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -89.5525665, 71.2040558, -89.6735382, 71.3006516, -160.8532104, 160.8775787
1: -74.7011871, 63.2646866, -74.8050461, 63.3510971, -138.0522461, 138.0697174
2: -98.5215759, 64.5024261, -98.6563034, 64.5893326, -163.1109009, 163.1587219
3: -104.9309311, 55.1410675, -105.0731049, 55.2181587, -160.1490784, 160.2141571
4: -96.4746780, 74.3184357, -96.6044388, 74.4187164, -170.8933716, 170.9228821
5: -86.0872574, 67.1749725, -86.2035370, 67.2643814, -153.3516388, 153.3785095
6: -83.0447693, 79.1879578, -83.1557770, 79.2965851, -162.3413544, 162.3437347
7: -90.4422226, 76.0658035, -90.5638504, 76.1672058, -166.6094360, 166.6296539
8: -108.0334625, 73.8858643, -108.1830673, 73.9889526, -182.0224152, 182.0689392
9: -82.5474014, 80.9413605, -82.6571503, 81.0513077, -163.5986786, 163.5984955

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 147

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2911901, upper bound: 197.2891171
time: 7.72 seconds

## Relational analysis of NS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2877731, upper bound: 197.2853766
time: 7.17 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2793098, upper bound: 197.2765706
time: 8.47 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -97.2147751, 77.2356796, -89.6735382, 71.3006516, -168.5154266, 166.9091949
1: -81.0425339, 68.6085815, -74.8050461, 63.3510971, -144.3936157, 143.4136353
2: -106.9261398, 69.9183121, -98.6563034, 64.5893326, -171.5154724, 168.5746155
3: -113.8855362, 59.8404198, -105.0731049, 55.2181587, -169.1036835, 164.9135284
4: -104.6767960, 80.5266190, -96.6044388, 74.4187164, -179.0955200, 177.1310577
5: -93.3957367, 72.8210297, -86.2035370, 67.2643814, -160.6601257, 159.0245361
6: -90.0346146, 85.9285278, -83.1557770, 79.2965851, -169.3312073, 169.0843048
7: -98.1290894, 82.4125290, -90.5638504, 76.1672058, -174.2962952, 172.9763794
8: -117.2155457, 80.1579666, -108.1830673, 73.9889526, -191.2044983, 188.3410339
9: -89.5210342, 87.8428268, -82.6571503, 81.0513077, -170.5723419, 170.4999695

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 147

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2791102, upper bound: 197.2767450
time: 8.10 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2793098, upper bound: 197.2770590
time: 7.36 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -89.5525665, 71.2040558, -97.3372955, 77.3335419, -166.8861084, 168.5413513
1: -74.7011871, 63.2646866, -81.1477661, 68.6961136, -143.3972473, 144.4124298
2: -98.5215759, 64.5024261, -107.0625916, 70.0063400, -168.5279236, 171.5650177
3: -104.9309311, 55.1410675, -114.0295410, 59.9184914, -164.8493958, 169.1705933
4: -96.4746780, 74.3184357, -104.8082352, 80.6281815, -177.1028595, 179.1266479
5: -86.0872574, 67.1749725, -93.5135345, 72.9115753, -158.9988251, 160.6885071
6: -83.0447693, 79.1879578, -90.1470566, 86.0385056, -169.0832825, 169.3350220
7: -90.4422226, 76.0658035, -98.2523117, 82.5152283, -172.9574585, 174.3181152
8: -108.0334625, 73.8858643, -117.3670349, 80.2623596, -188.2958221, 191.2528992
9: -82.5474014, 80.9413605, -89.6321716, 87.9542084, -170.5015717, 170.5735168

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 147

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2911901, upper bound: 197.3161191
time: 7.48 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2877731, upper bound: 197.3105322
time: 7.12 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2793098, upper bound: 197.3074154
time: 7.57 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -97.2147751, 77.2356796, -97.3372955, 77.3335419, -174.5483093, 174.5729675
1: -81.0425339, 68.6085815, -81.1477661, 68.6961136, -149.7386169, 149.7563477
2: -106.9261398, 69.9183121, -107.0625916, 70.0063400, -176.9324799, 176.9808960
3: -113.8855362, 59.8404198, -114.0295410, 59.9184914, -173.8039856, 173.8699493
4: -104.6767960, 80.5266190, -104.8082352, 80.6281815, -185.3049774, 185.3348389
5: -93.3957367, 72.8210297, -93.5135345, 72.9115753, -166.3073120, 166.3345490
6: -90.0346146, 85.9285278, -90.1470566, 86.0385056, -176.0731201, 176.0755768
7: -98.1290894, 82.4125290, -98.2523117, 82.5152283, -180.6443176, 180.6648407
8: -117.2155457, 80.1579666, -117.3670349, 80.2623596, -197.4779053, 197.5249939
9: -89.5210342, 87.8428268, -89.6321716, 87.9542084, -177.4752502, 177.4749908

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 147

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 64

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 64

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2911901, upper bound: 197.3277717
time: 8.14 seconds

## Relational analysis of NS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2877731, upper bound: 197.3206032
time: 6.51 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2793098, upper bound: 197.3163158
time: 8.28 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -96.6377411, 76.8255386, -89.2917404, 70.9961243, -167.6338501, 166.1172485
1: -80.4424744, 68.1649094, -74.4775391, 63.0788956, -143.5213623, 142.6424561
2: -106.2736511, 69.4814224, -98.2313995, 64.3152390, -170.5888977, 167.7128296
3: -113.2210770, 59.2589722, -104.6246338, 54.9750938, -168.1961670, 163.8836060
4: -104.1916580, 80.0082245, -96.1946945, 74.1022110, -178.2938690, 176.2029114
5: -92.9357910, 72.4521179, -85.8364258, 66.9825439, -159.9183197, 158.2885284
6: -89.6652679, 85.4194107, -82.8051987, 78.9544144, -168.6196594, 168.2245941
7: -97.5261002, 82.0366669, -90.1796188, 75.8473663, -173.3734589, 172.2162781
8: -116.3683319, 79.4070282, -107.7117004, 73.6641464, -190.0324707, 187.1187134
9: -89.0305252, 87.2649460, -82.3107758, 80.7047958, -169.7353058, 169.5757141

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 147

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2533819, upper bound: 197.2530347
time: 5.51 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
time: 5.90 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -96.6377411, 76.8255386, -96.9607849, 77.0332108, -173.6709290, 173.7862854
1: -80.4424744, 68.1649094, -80.8248596, 68.4277496, -148.8702240, 148.9897766
2: -106.2736511, 69.4814224, -106.6436539, 69.7360382, -176.0096893, 176.1250763
3: -113.2210770, 59.2589722, -113.5871353, 59.6788177, -172.8998871, 172.8460999
4: -104.1916580, 80.0082245, -104.4041290, 80.3161163, -184.5077820, 184.4123383
5: -92.9357910, 72.4521179, -93.1514206, 72.6335678, -165.5693359, 165.6035461
6: -89.6652679, 85.4194107, -89.8013306, 85.7010574, -175.3663330, 175.2207336
7: -97.5261002, 82.0366669, -97.8735199, 82.1997375, -179.7258148, 179.9101868
8: -116.3683319, 79.4070282, -116.9023285, 79.9420471, -196.3103790, 196.3093262
9: -89.0305252, 87.2649460, -89.2905579, 87.6124954, -176.6430206, 176.5554810

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 147

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2530347, upper bound: 197.2600997
time: 5.15 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2592019
time: 4.77 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -104.2336349, 82.8083878, -89.2917404, 70.9961243, -175.2297668, 172.1001282
1: -86.7353592, 73.4676285, -74.4775391, 63.0788956, -149.8142548, 147.9451447
2: -114.6076736, 74.8536148, -98.2313995, 64.3152390, -178.9228973, 173.0850220
3: -122.0956421, 63.9222336, -104.6246338, 54.9750938, -177.0707245, 168.5468750
4: -112.3205109, 86.1664200, -96.1946945, 74.1022110, -186.4227142, 182.3611145
5: -100.1814423, 78.0530777, -85.8364258, 66.9825439, -167.1639862, 163.8894806
6: -96.5953064, 92.1027374, -82.8051987, 78.9544144, -175.5497131, 174.9079285
7: -105.1498260, 88.3351059, -90.1796188, 75.8473663, -180.9971771, 178.5147247
8: -125.4759293, 85.6256790, -107.7117004, 73.6641464, -199.1400604, 193.3373718
9: -95.9430618, 94.1088257, -82.3107758, 80.7047958, -176.6478271, 176.4196014

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 147

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2600997, upper bound: 197.2588313
time: 7.93 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
time: 7.85 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -104.2336349, 82.8083878, -96.9607849, 77.0332108, -181.2668304, 179.7691650
1: -86.7353592, 73.4676285, -80.8248596, 68.4277496, -155.1631165, 154.2924805
2: -114.6076736, 74.8536148, -106.6436539, 69.7360382, -184.3437195, 181.4972687
3: -122.0956421, 63.9222336, -113.5871353, 59.6788177, -181.7744446, 177.5093689
4: -112.3205109, 86.1664200, -104.4041290, 80.3161163, -192.6366119, 190.5705566
5: -100.1814423, 78.0530777, -93.1514206, 72.6335678, -172.8150024, 171.2044983
6: -96.5953064, 92.1027374, -89.8013306, 85.7010574, -182.2963562, 181.9040680
7: -105.1498260, 88.3351059, -97.8735199, 82.1997375, -187.3495636, 186.2086029
8: -125.4759293, 85.6256790, -116.9023285, 79.9420471, -205.4179688, 202.5279999
9: -95.9430618, 94.1088257, -89.2905579, 87.6124954, -183.5555420, 183.3993683

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: A, layer: 1, pos: 122

### Candidate
type: B, layer: 1, pos: 122

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 93

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 105

### Candidate
type: B, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 105

### Candidate
type: A, layer: 1, pos: 226

### Candidate
type: B, layer: 1, pos: 226

### Candidate
type: A, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 123

### Candidate
type: A, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 199

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 114

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of NS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 147

### Candidate
type: B, layer: 1, pos: 213

### Candidate
type: A, layer: 1, pos: 64

## Relational analysis of NS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

### Candidate
type: B, layer: 1, pos: 64

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2600997, upper bound: 197.2985460
time: 5.12 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2976648
time: 5.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 21.58 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2877731, upper bound: 197.2853766
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2793098, upper bound: 197.2765706
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2791102, upper bound: 197.2767450
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2793098, upper bound: 197.2770590
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2877731, upper bound: 197.3105322
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2793098, upper bound: 197.3074154
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2877731, upper bound: 197.3206032
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2793098, upper bound: 197.3163158
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2533819, upper bound: 197.2530347
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2530347, upper bound: 197.2600997
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2592019
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2600997, upper bound: 197.2588313
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2600997, upper bound: 197.2985460
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.58
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2976648

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -82.9193649, 65.9278717, -72.7047958, 57.8058357, -140.7252045, 138.6326599
1: -69.0965576, 58.5933800, -60.4635239, 51.3974838, -120.4940414, 119.0569000
2: -91.1920929, 59.7627983, -79.9107513, 52.4657555, -143.6578522, 139.6735535
3: -97.1657104, 51.0774422, -85.2089005, 44.8221550, -141.9878693, 136.2863312
4: -89.3534927, 68.8549728, -78.3833923, 60.4433403, -149.7968292, 147.2383728
5: -79.6804123, 62.2048607, -69.8161697, 54.5529480, -134.2333221, 132.0210266
6: -76.9273834, 73.3195801, -67.5055771, 64.2831879, -141.2105408, 140.8251495
7: -83.8051453, 70.4440536, -73.5867233, 61.7900963, -145.5952454, 144.0307770
8: -100.0798340, 68.4294128, -87.8328934, 60.0266914, -160.1065063, 156.2622986
9: -76.5014267, 75.0167084, -67.1909103, 65.8934555, -142.3948822, 142.2076111

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2763572
time: 8.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2765706
time: 7.25 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -83.1415405, 66.1166611, -77.6767426, 61.7579193, -144.8994446, 143.7933960
1: -69.2981415, 58.7532425, -64.6774521, 54.8932190, -124.1913452, 123.4306946
2: -91.4407120, 59.9304657, -85.3847885, 56.0016632, -147.4423828, 145.3152466
3: -97.4258575, 51.2121468, -91.0172195, 47.8797264, -145.3055267, 142.2293549
4: -89.5974197, 69.0503540, -83.7531128, 64.5401611, -154.1375732, 152.8034363
5: -79.9139633, 62.3932648, -74.6309280, 58.3003998, -138.2143555, 137.0241699
6: -77.1476135, 73.5213013, -72.1113892, 68.6979904, -145.8456116, 145.6326599
7: -84.0268784, 70.6619186, -78.5636597, 66.0279617, -150.0548401, 149.2255707
8: -100.3437119, 68.6128998, -93.8038864, 64.1098175, -164.4535065, 162.4167786
9: -76.7059784, 75.2177277, -71.7370911, 70.3305511, -147.0365295, 146.9548035

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2763566
time: 7.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2765712
time: 7.92 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -80.2749786, 63.7588921, -83.0401688, 66.0243301, -146.2993011, 146.7990112
1: -66.7267227, 56.6734543, -69.2002563, 58.6796532, -125.4063721, 125.8737106
2: -88.2044830, 57.8142815, -91.3265915, 59.8495941, -148.0540771, 149.1408691
3: -94.0562286, 49.4596291, -97.3076401, 51.1544151, -145.2106171, 146.7672729
4: -86.4878387, 66.5736542, -89.4830704, 68.9550934, -155.4429169, 156.0567169
5: -77.0349808, 60.1297531, -79.7964859, 62.2941170, -139.3291016, 139.9262238
6: -74.4103622, 70.9377213, -77.0382309, 73.4280548, -147.8384094, 147.9759521
7: -81.1771393, 68.0567474, -83.9265671, 70.5452957, -151.7224274, 151.9833069
8: -96.8985596, 66.2172699, -100.2292023, 68.5324020, -165.4309540, 166.4464722
9: -74.0792313, 72.7125626, -76.6110535, 75.1265106, -149.2057190, 149.3236084

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2796280, upper bound: 197.2767091
time: 7.63 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2796280, upper bound: 197.2767097
time: 7.37 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -85.2047882, 67.6884842, -83.2623138, 66.2130966, -151.4178772, 150.9507751
1: -70.9151993, 60.1418533, -69.4018173, 58.8395462, -129.7547455, 129.5436554
2: -93.6381989, 61.3152580, -91.5752411, 60.0172577, -153.6554260, 152.8905029
3: -99.7982941, 52.4891243, -97.5677948, 51.2891235, -151.0874023, 150.0569153
4: -91.8077927, 70.6358185, -89.7270050, 69.1504745, -160.9582672, 160.3627930
5: -81.8091888, 63.8485909, -80.0300293, 62.4825211, -144.2917175, 143.8786163
6: -78.9730072, 75.3209763, -77.2584915, 73.6297531, -152.6027527, 152.5794678
7: -86.1098175, 72.2623520, -84.1483307, 70.7631760, -156.8729858, 156.4106750
8: -102.8187408, 70.2614975, -100.4930954, 68.7159042, -171.5346375, 170.7545929
9: -78.5861969, 77.1176605, -76.8156204, 75.3275681, -153.9137573, 153.9332886

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2800914, upper bound: 197.2770590
time: 7.73 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2800914, upper bound: 197.2770590
time: 7.80 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -82.9193649, 65.9278717, -80.3973465, 63.8565750, -146.7759399, 146.3252258
1: -69.0965576, 58.5933800, -66.8317795, 56.7608490, -125.8573990, 125.4251556
2: -91.1920929, 59.7627983, -88.3406830, 57.9022598, -149.0943298, 148.1034851
3: -97.1657104, 51.0774422, -94.2000122, 49.5375557, -146.7032623, 145.2774506
4: -89.3534927, 68.8549728, -86.6191101, 66.6750336, -156.0285339, 155.4740906
5: -79.6804123, 62.2048607, -77.1525803, 60.2202148, -139.9006042, 139.3574371
6: -76.9273834, 73.3195801, -74.5226669, 71.0475845, -147.9749451, 147.8422241
7: -83.8051453, 70.4440536, -81.3001480, 68.1593018, -151.9644470, 151.7441864
8: -100.0798340, 68.4294128, -97.0498199, 66.3215714, -166.4013977, 165.4792328
9: -76.5014267, 75.0167084, -74.1902847, 72.8238068, -149.3252258, 149.2070007

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074099
time: 18.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074154
time: 6.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -83.1415405, 66.1166611, -85.3263931, 67.7856674, -150.9272156, 151.4430389
1: -69.2981415, 58.7532425, -71.0195770, 60.2287750, -129.5269165, 129.7728271
2: -91.4407120, 59.9304657, -93.7736588, 61.4026871, -152.8433990, 153.7041321
3: -97.4258575, 51.2121468, -99.9412537, 52.5666275, -149.9924469, 151.1533813
4: -89.5974197, 69.0503540, -91.9383087, 70.7366791, -160.3341064, 160.9886475
5: -79.9139633, 62.3932648, -81.9261246, 63.9385414, -143.8525085, 144.3193817
6: -77.1476135, 73.5213013, -79.0847244, 75.4301834, -152.5777893, 152.6060028
7: -84.0268784, 70.6619186, -86.2321548, 72.3643341, -156.3912048, 156.8940735
8: -100.3437119, 68.6128998, -102.9691925, 70.3652420, -170.7089386, 171.5820923
9: -76.7059784, 75.2177277, -78.6966400, 77.2283401, -153.9343262, 153.9143677

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074099
time: 7.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074154
time: 6.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -90.6046829, 71.9763412, -80.3973465, 63.8565750, -154.4612274, 152.3736877
1: -75.4552231, 63.9508133, -66.8317795, 56.7608490, -132.2160645, 130.7825775
2: -99.6190948, 65.1949463, -88.3406830, 57.9022598, -157.5213623, 153.5356293
3: -106.1469727, 55.7906113, -94.2000122, 49.5375557, -155.6844940, 149.9906311
4: -97.5807190, 75.0806198, -86.6191101, 66.6750336, -164.2557526, 161.6997223
5: -87.0091400, 67.8673859, -77.1525803, 60.2202148, -147.2293549, 145.0199585
6: -83.9372787, 80.0790787, -74.5226669, 71.0475845, -154.9848328, 154.6017151
7: -91.5147934, 76.8082428, -81.3001480, 68.1593018, -159.6740875, 158.1083832
8: -109.2882462, 74.7193375, -97.0498199, 66.3215714, -175.6098175, 171.7691345
9: -83.4960175, 81.9395905, -74.1902847, 72.8238068, -156.3198242, 156.1298828

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3159153
time: 7.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3163158
time: 8.12 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -90.7286377, 72.0919266, -85.3263931, 67.7856674, -158.5143127, 157.4182892
1: -75.5782852, 64.0458832, -71.0195770, 60.2287750, -135.8070526, 135.0654602
2: -99.7658310, 65.2948227, -93.7736588, 61.4026871, -161.1685181, 159.0684814
3: -106.2919540, 55.8656349, -99.9412537, 52.5666275, -158.8585815, 155.8068848
4: -97.7218628, 75.2007599, -91.9383087, 70.7366791, -168.4585419, 167.1390686
5: -87.1519928, 67.9847946, -81.9261246, 63.9385414, -151.0905304, 149.9109192
6: -84.0706253, 80.1972427, -79.0847244, 75.4301834, -159.5007782, 159.2819672
7: -91.6405334, 76.9473267, -86.2321548, 72.3643341, -164.0048676, 163.1794739
8: -109.4390869, 74.8297195, -102.9691925, 70.3652420, -179.8043060, 177.7988892
9: -83.6154709, 82.0550003, -78.6966400, 77.2283401, -160.8438110, 160.7516479

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3159153
time: 7.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3163158
time: 8.93 seconds

## BFS NS instance: NS_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -79.7141495, 63.3681717, -82.6567841, 65.7185974, -145.4327240, 146.0249634
1: -66.1414871, 56.2441940, -68.8714600, 58.4063072, -124.5477905, 125.1156540
2: -87.5757675, 57.3901367, -90.8999176, 59.5743217, -147.1500854, 148.2900391
3: -93.4122238, 48.8841858, -96.8573532, 50.9102745, -144.3224792, 145.7415466
4: -86.0292435, 66.0666122, -89.0716248, 68.6373062, -154.6665497, 155.1382446
5: -76.5973434, 59.7782669, -79.4279709, 62.0111465, -138.6084900, 139.2062378
6: -74.0631561, 70.4496384, -76.6862183, 73.0844269, -147.1475830, 147.1358643
7: -80.5937805, 67.7021484, -83.5407028, 70.2240982, -150.8178711, 151.2428589
8: -96.0770416, 65.4734268, -99.7559280, 68.2061310, -164.2831726, 165.2293549
9: -73.6004791, 72.1509933, -76.2631683, 74.7785339, -148.3789673, 148.4141388

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
time: 5.29 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
time: 5.46 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -84.7892303, 67.3975525, -82.8802490, 65.9083939, -150.6976318, 150.2778015
1: -70.4407196, 59.8052940, -69.0741425, 58.5672112, -129.0079346, 128.8794098
2: -93.1540298, 60.9866943, -91.1501236, 59.7429276, -152.8969574, 152.1368103
3: -99.3201294, 51.9998589, -97.1191711, 51.0458260, -150.3659210, 149.1190033
4: -91.4877701, 70.2410126, -89.3170395, 68.8338242, -160.3215790, 159.5580444
5: -81.4993668, 63.5953217, -79.6627502, 62.2005119, -143.6998749, 143.2580719
6: -78.7426910, 74.9474945, -76.9077072, 73.2873383, -152.0300293, 151.8551941
7: -85.6664429, 72.0145340, -83.7639618, 70.4431076, -156.1095428, 155.7784882
8: -102.1544876, 69.6273117, -100.0214386, 68.3907623, -170.5452576, 169.6487427
9: -78.2313309, 76.6769485, -76.4690170, 74.9808426, -153.2121429, 153.1459503

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
time: 5.08 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
time: 4.49 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -90.0526886, 71.5898514, -80.0159378, 63.5524673, -153.6051636, 151.6057892
1: -74.8795090, 63.5283737, -66.5047302, 56.4890633, -131.3685608, 130.0330811
2: -98.9986877, 64.7766800, -87.9164963, 57.6281891, -156.6268616, 152.6931763
3: -105.5114365, 55.2225723, -93.7519073, 49.2948341, -154.8062592, 148.9744568
4: -97.1244354, 74.5833893, -86.2097702, 66.3589935, -163.4833984, 160.7931519
5: -86.5769882, 67.5189819, -76.7858582, 59.9386520, -146.5156403, 144.3048248
6: -83.5941696, 79.5950851, -74.1724625, 70.7058105, -154.2999878, 153.7675476
7: -90.9370117, 76.4568100, -80.9165344, 67.8397751, -158.7767944, 157.3733521
8: -108.4733124, 73.9874802, -96.5789719, 65.9968643, -174.4701691, 170.5664368
9: -83.0267029, 81.3848190, -73.8441086, 72.4776230, -155.5043182, 155.2289276

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
time: 7.61 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
time: 7.23 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -90.2352142, 71.7443695, -84.9478073, 67.4835739, -157.7187653, 156.6921539
1: -75.0467834, 63.6596069, -70.6948166, 59.9588356, -135.0055847, 134.3543854
2: -99.2015762, 64.9151535, -93.3522949, 61.1306038, -160.3321686, 158.2674408
3: -105.7253418, 55.3322372, -99.4964066, 52.3255005, -158.0508423, 154.8286285
4: -97.3253250, 74.7456818, -91.5318527, 70.4228745, -167.7481995, 166.2775269
5: -86.7695618, 67.6755981, -81.5620193, 63.6589622, -150.4285126, 149.2376099
6: -83.7747421, 79.7601471, -78.7369843, 75.0907669, -158.8655090, 158.4971161
7: -91.1191940, 76.6389160, -85.8511734, 72.0469513, -163.1661072, 162.4900818
8: -108.6881409, 74.1381302, -102.5018616, 70.0429001, -178.7310333, 176.6399841
9: -83.1957169, 81.5472641, -78.3528595, 76.8845215, -160.0802307, 159.9001160

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
time: 6.52 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
time: 6.09 seconds

## BFS NS instance: NS_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -87.3775482, 69.4018402, -82.6567841, 65.7185974, -153.0961304, 152.0586243
1: -72.4931259, 61.5932693, -68.8714600, 58.4063072, -130.8994293, 130.4647217
2: -95.9823685, 62.8078728, -90.8999176, 59.5743217, -155.5566864, 153.7077942
3: -102.3660126, 53.5895424, -96.8573532, 50.9102745, -153.2762299, 150.4468842
4: -94.2318497, 72.2797318, -89.0716248, 68.6373062, -162.8691559, 161.3513489
5: -83.9043274, 65.4282150, -79.4279709, 62.0111465, -145.9154663, 144.8561859
6: -81.0539093, 77.1925888, -76.6862183, 73.0844269, -154.1383362, 153.8788147
7: -88.2848434, 74.0552902, -83.5407028, 70.2240982, -158.5089417, 157.5959930
8: -105.2617569, 71.7458115, -99.7559280, 68.2061310, -173.4678955, 171.5017395
9: -80.5766525, 79.0567856, -76.2631683, 74.7785339, -155.3551636, 155.3199310

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
time: 6.33 seconds

## Relational analysis of NS_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
time: 6.56 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -92.1755676, 73.2212753, -82.8802490, 65.9083939, -158.0839539, 156.1015167
1: -76.5663986, 64.9658051, -69.0741425, 58.5672112, -135.1336060, 134.0399017
2: -101.2643585, 66.2148666, -91.1501236, 59.7429276, -161.0072937, 157.3649902
3: -107.9520569, 56.5333786, -97.1191711, 51.0458260, -158.9978790, 153.6525116
4: -99.3991699, 76.2324219, -89.3170395, 68.8338242, -168.2329254, 165.5494537
5: -88.5494232, 69.0441895, -79.6627502, 62.2005119, -150.7499390, 148.7069397
6: -85.4870911, 81.4509430, -76.9077072, 73.2873383, -158.7744141, 158.3586426
7: -93.0809937, 78.1447296, -83.7639618, 70.4431076, -163.5240936, 161.9086914
8: -111.0176239, 75.6798019, -100.0214386, 68.3907623, -179.4083862, 175.7012329
9: -84.9616623, 83.3370743, -76.4690170, 74.9808426, -159.9424744, 159.8060913

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 93

### Candidate
type: A, layer: 1, pos: 187

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
time: 6.44 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
time: 6.87 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -87.3775482, 69.4018402, -90.3488312, 71.7724304, -159.1499786, 159.7506714
1: -72.4931259, 61.5932693, -75.2359924, 63.7686882, -136.2618103, 136.8292542
2: -95.9823685, 62.8078728, -99.3345871, 65.0112534, -160.9936066, 162.1424561
3: -102.3660126, 53.5895424, -105.8464661, 55.6278687, -157.9938354, 159.4359894
4: -94.2318497, 72.2797318, -97.3060837, 74.8685684, -169.1003876, 169.5858154
5: -83.9043274, 65.4282150, -86.7631302, 67.6785812, -151.5829163, 152.1913452
6: -81.0539093, 77.1925888, -83.7023849, 79.8500137, -160.9039307, 160.8949738
7: -88.2848434, 74.0552902, -91.2573395, 76.5939102, -164.8787384, 165.3126221
8: -105.2617569, 71.7458115, -108.9727249, 74.5017624, -179.7635193, 180.7185364
9: -80.5766525, 79.0567856, -83.2638474, 81.7075119, -162.2841644, 162.3206329

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648
time: 6.01 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648
time: 5.47 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -92.1755676, 73.2212753, -90.4739456, 71.8888702, -164.0644379, 163.6952057
1: -76.5663986, 64.9658051, -75.3599548, 63.8645592, -140.4309387, 140.3257294
2: -101.2643585, 66.2148666, -99.4824524, 65.1120148, -166.3763733, 165.6973267
3: -107.9520569, 56.5333786, -105.9928741, 55.7036400, -163.6557007, 162.5262451
4: -99.3991699, 76.2324219, -97.4484634, 74.9896240, -174.3887482, 173.6808472
5: -88.5494232, 69.0441895, -86.9070435, 67.7968750, -156.3462982, 155.9512329
6: -85.4870911, 81.4509430, -83.8367004, 79.9691467, -165.4562073, 165.2876434
7: -93.0809937, 78.1447296, -91.3842316, 76.7339172, -169.8148956, 169.5289612
8: -111.0176239, 75.6798019, -109.1250381, 74.6131897, -185.6307983, 184.8048401
9: -84.9616623, 83.3370743, -83.3843384, 81.8240814, -166.7857361, 166.7214050

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648
time: 5.65 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648
time: 5.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 12.34 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2763572
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2765706
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2763566
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2790218, upper bound: 197.2765712
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2796280, upper bound: 197.2767091
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2796280, upper bound: 197.2767097
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2800914, upper bound: 197.2770590
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2800914, upper bound: 197.2770590
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074099
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074154
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074099
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3084309, upper bound: 197.3074154
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3159153
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3163158
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3159153
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.3174625, upper bound: 197.3163158
NS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
NS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
NS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
NS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2580457, upper bound: 197.2592019
NS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
NS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
NS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
NS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2592019, upper bound: 197.2580457
NS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648
NS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648
NS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648
NS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 12.34
Output dim: 4, lower bound: -197.2976680, upper bound: 197.2976648

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 9.02 + 600.52 = 609.54 seconds
