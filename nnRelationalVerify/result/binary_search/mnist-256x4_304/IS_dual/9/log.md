## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 197.2433907684
Search space: {k/256 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 1.40 + 8.25 = 9.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4409435, upper bound: 197.4409435


# Binary Search by BASE starts (time budget: 1990.35 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=198.953369140625
rel_dist={4: [-197.44087218970873, 197.4408721892934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=198.953369140625
rel_dist={4: [-197.44083159618555, 197.44083163160866]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=198.953369140625
rel_dist={4: [-197.4407374020123, 197.4407374020123]}

## Binary Search Result
Binary search time: 32.90 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1957.45 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3557765, upper bound: 197.3540236
time: 7.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3232616, upper bound: 197.3232616
time: 4.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.10
Output dim: 4, lower bound: -197.3557765, upper bound: 197.3540236
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.10
Output dim: 4, lower bound: -197.3232616, upper bound: 197.3232616

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2890287, upper bound: 197.2891856
time: 11.28 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3533695, upper bound: 197.3518570
time: 7.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -111.3679199, 88.4211121, -104.3641129, 82.8620148, -194.2299347, 192.7851868
1: -92.6760406, 78.4514008, -86.9970474, 73.6025696, -166.2786102, 165.4484253
2: -122.4418869, 79.8888092, -114.7780228, 74.9635010, -197.4053497, 194.6668396
3: -130.4202118, 68.2964859, -122.2262726, 64.2218552, -194.6420593, 190.5227661
4: -119.9568024, 91.9793015, -112.3292465, 86.3542862, -206.3110962, 204.3085480
5: -107.0011215, 83.3186722, -100.2296448, 78.0977097, -185.0988312, 183.5482941
6: -103.0942459, 98.3950500, -96.5462112, 92.2351456, -195.3293915, 194.9412384
7: -112.2794952, 94.2552719, -105.2733154, 88.3454361, -200.6249390, 199.5285950
8: -134.0392761, 91.4946136, -125.7949600, 86.0409012, -220.0801697, 217.2895660
9: -102.4540939, 100.5486450, -96.0442200, 94.2965698, -196.7506714, 196.5928650

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2918228, upper bound: 197.2898510
time: 6.22 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3205236, upper bound: 197.3205236
time: 4.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 67.62 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 67.62
Output dim: 4, lower bound: -197.2890287, upper bound: 197.2891856
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 67.62
Output dim: 4, lower bound: -197.3533695, upper bound: 197.3518570
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 67.62
Output dim: 4, lower bound: -197.2918228, upper bound: 197.2898510
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 67.62
Output dim: 4, lower bound: -197.3205236, upper bound: 197.3205236

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -91.8739243, 72.8921738, -101.8935242, 80.8960114, -172.7699280, 174.7857056
1: -76.1764832, 64.6233978, -84.8809052, 71.8421097, -148.0185852, 149.5043030
2: -100.9808655, 66.0587692, -112.0536270, 73.2083817, -174.1892242, 178.1123962
3: -107.7184753, 56.4441032, -119.3473587, 62.6921082, -170.4105835, 175.7914581
4: -98.8170853, 75.8664780, -109.6574326, 84.2993546, -183.1164093, 185.5239105
5: -88.2734680, 68.7790222, -97.8613663, 76.2587891, -164.5322571, 166.6403809
6: -84.9869690, 81.0246201, -94.2609634, 90.0292435, -175.0162048, 175.2855835
7: -92.6945877, 77.8304138, -102.7894974, 86.2687302, -178.9633026, 180.6199036
8: -110.4494324, 75.2781372, -122.7793274, 83.9468079, -194.3962250, 198.0574646
9: -84.5107498, 82.7721939, -93.7723618, 92.0397263, -176.5504761, 176.5445557

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2522030, upper bound: 197.2523864
time: 7.12 seconds

## Relational analysis of IS_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2752543, upper bound: 197.2748541
time: 7.18 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2781758, upper bound: 197.2777825
time: 6.13 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -101.6408539, 80.7009125, -104.5059052, 82.9751511, -184.6159973, 185.2068176
1: -84.6728668, 71.6667252, -87.1186676, 73.7036362, -158.3764954, 158.7854004
2: -111.7772598, 73.0326462, -114.9358521, 75.0652771, -186.8425293, 187.9684753
3: -119.0536499, 62.5364189, -122.3928833, 64.3120346, -183.3656921, 184.9292908
4: -109.3885498, 84.0957870, -112.4815369, 86.4718399, -195.8603516, 196.5773315
5: -97.6223907, 76.0722656, -100.3661041, 78.2024307, -175.8248138, 176.4383698
6: -94.0302200, 89.8075027, -96.6764297, 92.3622818, -186.3925018, 186.4839020
7: -102.5391312, 86.0565796, -105.4159775, 88.4643326, -191.0034180, 191.4725647
8: -122.4789429, 83.7473145, -125.9699860, 86.1614532, -208.6403961, 209.7173004
9: -93.5436859, 91.8160172, -96.1728745, 94.4252930, -187.9689636, 187.9888916

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3342049, upper bound: 197.3329665
time: 7.31 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2963531, upper bound: 197.2957237
time: 6.62 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3492069, upper bound: 197.3476198
time: 6.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -110.4277802, 87.6810760, -89.5322647, 71.1879807, -181.6157532, 177.2133484
1: -91.8948593, 77.7944870, -74.6838608, 63.2503777, -155.1452332, 152.4783478
2: -121.4089661, 79.2240372, -98.4990997, 64.4879150, -185.8968658, 177.7231445
3: -129.3217621, 67.7188416, -104.9071655, 55.1282234, -184.4499817, 172.6260071
4: -118.9508743, 91.2149124, -96.4528351, 74.3016129, -193.2524567, 187.6677551
5: -106.1028366, 82.6245575, -86.0676956, 67.1601028, -173.2629242, 168.6922607
6: -102.2384567, 97.5663986, -83.0260544, 79.1699829, -181.4084167, 180.5924225
7: -111.3373871, 93.4753189, -90.4216919, 76.0488663, -187.3862305, 183.8970032
8: -132.9116211, 90.7217484, -108.0086823, 73.8687820, -206.7803955, 198.7304077
9: -101.5964584, 99.6992416, -82.5290070, 80.9231033, -182.5195465, 182.2282257

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 179

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2786866, upper bound: 197.2786866
time: 5.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2786866, upper bound: 197.2898508
time: 5.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -111.0866776, 88.1998749, -97.1979599, 77.2223892, -188.3090668, 185.3978271
1: -92.4419937, 78.2550430, -81.0282745, 68.5967941, -161.0387573, 159.2833099
2: -122.1331177, 79.6903915, -106.9075546, 69.9063187, -192.0394287, 186.5979462
3: -130.0920105, 68.1241379, -113.8658218, 59.8297997, -189.9218140, 181.9899292
4: -119.6558228, 91.7501907, -104.6586838, 80.5126953, -200.1684875, 196.4088593
5: -106.7323303, 83.1111374, -93.3795319, 72.8087006, -179.5410309, 176.4906616
6: -102.8380890, 98.1470490, -90.0191269, 85.9136505, -188.7517395, 188.1661682
7: -111.9984512, 94.0219574, -98.1121216, 82.3984756, -194.3969269, 192.1340790
8: -133.7019043, 91.2634201, -117.1950455, 80.1438065, -213.8457031, 208.4584656
9: -102.1974640, 100.2948532, -89.5057526, 87.8277588, -190.0252228, 189.8005981

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2898510, upper bound: 197.2918228
time: 5.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2898510, upper bound: 197.3205236
time: 5.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 45.49 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.2752543, upper bound: 197.2748541
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.2781758, upper bound: 197.2777825
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.2963531, upper bound: 197.2957237
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.3492069, upper bound: 197.3476198
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.2786866, upper bound: 197.2786866
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.2786866, upper bound: 197.2898508
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.2898510, upper bound: 197.2918228
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 45.49
Output dim: 4, lower bound: -197.2898510, upper bound: 197.3205236

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -90.3687973, 71.7034531, -83.8115005, 66.6172867, -156.9860687, 155.5149536
1: -74.8961563, 63.5579567, -69.5198669, 59.0565338, -133.9526672, 133.0778198
2: -99.3179932, 64.9834442, -92.0908508, 60.3029709, -159.6209259, 157.0742798
3: -105.9538879, 55.5136719, -98.1526413, 51.5244217, -157.4783020, 153.6663208
4: -97.2003326, 74.6242142, -90.2332687, 69.3863907, -166.5867310, 164.8574829
5: -86.8348770, 67.6587982, -80.5758438, 62.8073273, -149.6421814, 148.2346344
6: -83.6040726, 79.6858521, -77.6426773, 73.9610367, -157.5650940, 157.3285217
7: -91.1757965, 76.5632629, -84.5582352, 71.0538483, -162.2296295, 161.1214905
8: -108.6288605, 74.0311432, -100.9249878, 68.9810257, -177.6098785, 174.9561310
9: -83.1192017, 81.3948288, -77.0785294, 75.5277252, -158.6469116, 158.4733582

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2586631, upper bound: 197.2586554
time: 7.42 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2696684, upper bound: 197.2691571
time: 6.54 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -91.2479706, 72.3982315, -90.3565063, 71.7921829, -163.0401611, 162.7547302
1: -75.6446609, 64.1808701, -75.0797577, 63.6910744, -139.3357086, 139.2606201
2: -100.2893829, 65.6127701, -99.3124313, 64.9965439, -165.2859192, 164.9252014
3: -106.9851303, 56.0575638, -105.8278503, 55.5661621, -162.5513000, 161.8854065
4: -98.1451797, 75.3509369, -97.2722168, 74.7969742, -172.9421387, 172.6231384
5: -87.6754532, 68.3138733, -86.8372498, 67.6869507, -155.3623962, 155.1510925
6: -84.4129868, 80.4680176, -83.6841278, 79.7734604, -164.1864471, 164.1521149
7: -92.0643158, 77.3052979, -91.1734543, 76.5941620, -168.6584778, 168.4787598
8: -109.6932907, 74.7594376, -108.8498077, 74.3909683, -184.0842438, 183.6092529
9: -83.9343491, 82.2019730, -83.1561890, 81.5358963, -165.4701996, 165.3581543

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -197.2407377, upper bound: 197.2404197
time: 7.65 seconds

## Relational analysis of IS_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2723230, upper bound: 197.2722113
time: 7.94 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2723230, upper bound: 197.2777825
time: 8.14 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -77.1329956, 61.1410408, -100.2639008, 79.5944519, -156.7274475, 161.4049377
1: -63.8172798, 54.2694740, -83.5150375, 70.6974564, -134.5147247, 137.7845154
2: -84.5995026, 55.5621223, -110.2355957, 72.0445023, -156.6440125, 165.7977142
3: -90.4786453, 47.4248466, -117.4377365, 61.6957703, -152.1743927, 164.8625488
4: -82.9020157, 63.7134171, -107.8952103, 82.9497681, -165.8517761, 171.6086121
5: -74.1265182, 57.7485733, -96.2967300, 75.0339584, -149.1604767, 154.0453033
6: -71.5127487, 67.9392929, -92.7775345, 88.5808029, -160.0935516, 160.7167816
7: -77.9210815, 65.4739838, -101.1543808, 84.9008255, -162.8218994, 166.6283417
8: -92.5755539, 63.0208092, -120.8039017, 82.5889969, -175.1645508, 183.8247070
9: -70.9770508, 69.4168472, -92.2732697, 90.5586166, -161.5356750, 161.6901245

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2532550, upper bound: 197.2524778
time: 7.06 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2909577, upper bound: 197.2906993
time: 7.17 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -96.2660522, 76.4247437, -104.5059052, 82.9751511, -179.2411804, 180.9306183
1: -80.1168365, 67.8638000, -87.1186676, 73.7036362, -153.8204651, 154.9824677
2: -105.8319702, 69.2018814, -114.9358521, 75.0652771, -180.8972321, 184.1376953
3: -112.7743301, 59.2225838, -122.3928833, 64.3120346, -177.0863647, 181.6154633
4: -103.5890274, 79.6568375, -112.4815369, 86.4718399, -190.0608368, 192.1383667
5: -92.4655304, 72.0620575, -100.3661041, 78.2024307, -170.6679688, 172.4281616
6: -89.0970078, 85.0272980, -96.6764297, 92.3622818, -181.4592743, 181.7036743
7: -97.1406479, 81.5405807, -105.4159775, 88.4643326, -185.6049347, 186.9565582
8: -115.9611969, 79.2556458, -125.9699860, 86.1614532, -202.1226501, 205.2256317
9: -88.6095581, 86.9345245, -96.1728745, 94.4252930, -183.0348511, 183.1073608

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3295937, upper bound: 197.3280138
time: 8.00 seconds

## Relational analysis of IS_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3184890, upper bound: 197.3164205
time: 7.68 seconds

## Relational analysis of IS_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3491320, upper bound: 197.3476198
time: 7.11 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -96.6377411, 76.8255386, -89.5322647, 71.1879807, -167.8257141, 166.3577881
1: -80.4424744, 68.1649094, -74.6838608, 63.2503777, -143.6928558, 142.8487701
2: -106.2736511, 69.4814224, -98.4990997, 64.4879150, -170.7615662, 167.9805145
3: -113.2210770, 59.2589722, -104.9071655, 55.1282234, -168.3493042, 164.1661377
4: -104.1916580, 80.0082245, -96.4528351, 74.3016129, -178.4932709, 176.4610443
5: -92.9357910, 72.4521179, -86.0676956, 67.1601028, -160.0958862, 158.5198059
6: -89.6652679, 85.4194107, -83.0260544, 79.1699829, -168.8352356, 168.4454498
7: -97.5261002, 82.0366669, -90.4216919, 76.0488663, -173.5749512, 172.4583588
8: -116.3683319, 79.4070282, -108.0086823, 73.8687820, -190.2371216, 187.4156799
9: -89.0305252, 87.2649460, -82.5290070, 80.9231033, -169.9535980, 169.7939301

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=198.953369140625
rel_dist={4: [-197.44087218970873, 197.4408721892934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3406430, upper bound: 197.3402623
time: 8.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3232398, upper bound: 197.3232398
time: 5.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.96
Output dim: 4, lower bound: -197.3406430, upper bound: 197.3402623
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.96
Output dim: 4, lower bound: -197.3232398, upper bound: 197.3232398

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2965964, upper bound: 197.2958543
time: 8.25 seconds

## Relational analysis of IS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2929954, upper bound: 197.2923500
time: 8.07 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3318468, upper bound: 197.3311224
time: 6.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2849843, upper bound: 197.2859300
time: 7.57 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3205095, upper bound: 197.3205095
time: 4.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 61.26 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 61.26
Output dim: 4, lower bound: -197.2929954, upper bound: 197.2923500
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 61.26
Output dim: 4, lower bound: -197.3318468, upper bound: 197.3311224
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 61.26
Output dim: 4, lower bound: -197.2849843, upper bound: 197.2859300
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 61.26
Output dim: 4, lower bound: -197.3205095, upper bound: 197.3205095

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -80.8864517, 64.3105927, -94.4778824, 75.0527802, -155.9392242, 158.7884216
1: -67.0065460, 56.9711151, -78.5875778, 66.6046371, -133.6111755, 135.5586853
2: -88.8122025, 58.2364464, -103.8538284, 67.9197006, -156.7319031, 162.0902557
3: -94.7161179, 49.6519814, -110.6412125, 58.0916252, -152.8077393, 160.2931976
4: -87.1010513, 67.0012970, -101.7033539, 78.2132492, -165.3143005, 168.7046509
5: -77.7540512, 60.6227112, -90.7620010, 70.7347641, -148.4887848, 151.3846893
6: -74.9265366, 71.3796844, -87.4449768, 83.4574738, -158.3840027, 158.8246460
7: -81.6639404, 68.5412827, -95.3363953, 80.0073013, -161.6712341, 163.8776855
8: -97.5491791, 66.7005157, -113.9116898, 77.9106979, -175.4598236, 180.6121979
9: -74.4086914, 72.9362183, -86.9414597, 85.3123779, -159.7210388, 159.8776855

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2783160, upper bound: 197.2775788
time: 7.71 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2802736, upper bound: 197.2796291
time: 8.05 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -98.3072662, 78.0776978, -103.3190689, 82.0377426, -180.3450012, 181.3967438
1: -81.8530426, 69.3214493, -86.1110535, 72.8649826, -154.7180023, 155.4324951
2: -108.0869751, 70.6636124, -113.6248474, 74.2228088, -182.3097839, 184.2884521
3: -115.1359100, 60.4771118, -121.0037766, 63.5783386, -178.7142487, 181.4808655
4: -105.8214951, 81.3723145, -111.2068405, 85.4957428, -191.3172150, 192.5791473
5: -94.4273376, 73.5907745, -99.2292480, 77.3193817, -171.7467194, 172.8200226
6: -90.9737091, 86.8606796, -95.5848389, 91.3092957, -182.2830048, 182.4455261
7: -99.1959381, 83.2428589, -104.2252655, 87.4646301, -186.6605682, 187.4681244
8: -118.5101395, 81.0615082, -124.5427856, 85.1862411, -203.6963806, 205.6042938
9: -90.4846420, 88.8111343, -95.0838470, 93.3505783, -183.8352203, 183.8949890

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B1
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
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2859818, upper bound: 197.2850123
time: 7.79 seconds

## Relational analysis of IS_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2949543, upper bound: 197.2920382
time: 7.52 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3296624, upper bound: 197.3284355
time: 7.25 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2786441
time: 4.66 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2859300
time: 4.76 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2859300, upper bound: 197.2849843
time: 6.98 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2859300, upper bound: 197.3205095
time: 7.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.28 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.2783160, upper bound: 197.2775788
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.2802736, upper bound: 197.2796291
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.2949543, upper bound: 197.2920382
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.3296624, upper bound: 197.3284355
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2786441
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.2786443, upper bound: 197.2859300
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.2859300, upper bound: 197.2849843
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 19.28
Output dim: 4, lower bound: -197.2859300, upper bound: 197.3205095

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -74.0810013, 58.9356155, -76.6217422, 60.9515915, -135.0325928, 135.5573578
1: -61.2161674, 52.1539383, -63.4117165, 53.9748917, -115.1910553, 115.5656586
2: -81.2931061, 53.3762627, -84.1380386, 55.1759491, -136.4690552, 137.5142517
3: -86.7408142, 45.4465523, -89.7169647, 47.0617676, -133.8025818, 135.1634979
4: -79.7906647, 61.3857841, -82.5269318, 63.4864273, -143.2770996, 143.9127197
5: -71.2504730, 55.5606956, -73.6931686, 57.4521446, -128.7026215, 129.2538300
6: -68.6736145, 65.3263168, -71.0386658, 67.5866470, -136.2602539, 136.3649902
7: -74.7982407, 62.8102341, -77.3357697, 64.9774780, -139.7756500, 140.1459961
8: -89.3196411, 61.0609779, -92.3272018, 63.1296921, -152.4493103, 153.3881836
9: -68.1151886, 66.7076569, -70.4512711, 69.0016937, -137.1168365, 137.1588898

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2783160, upper bound: 197.2775788
time: 8.96 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2783160, upper bound: 197.2775788
time: 8.75 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -76.4001465, 60.7696762, -82.9700470, 65.9717178, -142.3718567, 143.7397156
1: -63.1909256, 53.7990265, -68.8082962, 58.4711800, -121.6620865, 122.6073227
2: -83.8556213, 55.0393486, -91.1438293, 59.7275696, -143.5831604, 146.1831512
3: -89.4621582, 46.8823700, -97.1601028, 50.9847984, -140.4469452, 144.0424805
4: -82.2858353, 63.3067589, -89.3517685, 68.7346573, -151.0204926, 152.6585236
5: -73.4695129, 57.2903938, -79.7674103, 62.1853981, -135.6549072, 137.0577850
6: -70.8133621, 67.3899307, -76.8952789, 73.2273712, -144.0407410, 144.2852020
7: -77.1473007, 64.7758026, -83.7526245, 70.3557205, -147.5030212, 148.5284271
8: -92.1313705, 62.9838562, -100.0155640, 68.3773041, -160.5086670, 162.9994049
9: -70.2750549, 68.8476562, -76.3487854, 74.8339539, -145.1090088, 145.1964417

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 249

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2794538, upper bound: 197.2788101
time: 8.19 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2794538, upper bound: 197.2796291
time: 9.16 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -93.0853043, 73.9691162, -88.4954834, 70.3701706, -163.4554749, 162.4645844
1: -77.5165405, 65.6752243, -73.8044662, 62.5183830, -140.0349274, 139.4796600
2: -102.3521271, 66.9744949, -97.3546982, 63.7526093, -166.1047363, 164.3291931
3: -109.0354004, 57.2713203, -103.6939697, 54.4896736, -163.5250702, 160.9652863
4: -100.2337723, 77.1282806, -95.3384094, 73.4497681, -173.6835327, 172.4666748
5: -89.4394913, 69.7372589, -85.0749741, 66.3879318, -155.8274078, 154.8122253
6: -86.2209015, 82.2592163, -82.0714951, 78.2511444, -164.4720306, 164.3307190
7: -93.9662781, 78.9132080, -89.3816452, 75.1746216, -169.1408997, 168.2948456
8: -112.2513885, 76.7722778, -106.7661514, 73.0211105, -185.2724457, 183.5384216
9: -85.7227020, 84.0955887, -81.5756683, 79.9844894, -165.7071838, 165.6712494

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
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
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2941735, upper bound: 197.2916268
time: 7.77 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2941735, upper bound: 197.2920382
time: 7.52 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -95.5275421, 75.8904419, -96.1492004, 76.3950653, -171.9225464, 172.0396271
1: -79.5379486, 67.3800888, -80.1389084, 67.8566513, -147.3945923, 147.5189972
2: -105.0342102, 68.7021332, -105.7502518, 69.1630783, -174.1972504, 174.4523773
3: -111.8934326, 58.7739983, -112.6391449, 59.1841125, -171.0775452, 171.4131012
4: -102.8459396, 79.1069717, -103.5320053, 79.6511993, -182.4971313, 182.6389618
5: -91.7710648, 71.5391846, -92.3755875, 72.0276184, -163.7986298, 163.9147644
6: -88.4421005, 84.4086380, -89.0543823, 84.9844055, -173.4264984, 173.4630127
7: -96.4181061, 80.9359207, -97.0605469, 81.5144653, -177.9325104, 177.9964447
8: -115.1737442, 78.7750549, -115.9382248, 79.2859802, -194.4597015, 194.7132721
9: -87.9484177, 86.3014832, -88.5418777, 86.8785553, -174.8269653, 174.8433533

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3200596, upper bound: 197.3189730
time: 7.84 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3200596, upper bound: 197.3284355
time: 6.91 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2533819, upper bound: 197.2530347
time: 5.85 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2507930
time: 6.27 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 226

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 226

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2530347, upper bound: 197.2600997
time: 5.52 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2507930, upper bound: 197.2592019
time: 5.07 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
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

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=198.953369140625
rel_dist={4: [-197.44083159618555, 197.44083163160866]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3290181, upper bound: 197.3291008
time: 10.60 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3231596, upper bound: 197.3231596
time: 5.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.42 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 16.42
Output dim: 4, lower bound: -197.3290181, upper bound: 197.3291008
IS_B2, status: Status.UNKNOWN, split count: 1, time: 16.42
Output dim: 4, lower bound: -197.3231596, upper bound: 197.3231596

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -104.5059052, 82.9751511, -104.3831787, 82.8770676, -187.3829651, 187.3583069
1: -87.1186676, 73.7036362, -87.0132828, 73.6159668, -160.7346344, 160.7169189
2: -114.9358521, 75.0652771, -114.7991562, 74.9771271, -189.9129333, 189.8644257
3: -122.3928833, 64.3120346, -122.2485809, 64.2338867, -186.6267548, 186.5606079
4: -112.4815369, 86.4718399, -112.3497925, 86.3700867, -198.8516235, 198.8216248
5: -100.3661041, 78.2024307, -100.2480621, 78.1116562, -178.4777527, 178.4505005
6: -96.6764297, 92.3622818, -96.5637970, 92.2520523, -188.9284515, 188.9260559
7: -105.4159775, 88.4643326, -105.2925644, 88.3614044, -193.7773743, 193.7568512
8: -125.9699860, 86.1614532, -125.8182297, 86.0569229, -212.0269012, 211.9796753
9: -96.1728745, 94.4252930, -96.0615463, 94.3137054, -190.4865723, 190.4868317

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

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
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 133

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3253311, upper bound: 197.3256712
time: 9.54 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3249350, upper bound: 197.3252364
time: 9.95 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -103.3304672, 82.0374908, -111.3679199, 88.4211121, -191.7515259, 193.4054108
1: -86.1104355, 72.8657074, -92.6760406, 78.4514008, -164.5618286, 165.5417480
2: -113.6277313, 74.2213516, -122.4418869, 79.8888092, -193.5165405, 196.6632385
3: -121.0117035, 63.5640030, -130.4202118, 68.2964859, -189.3081970, 193.9842072
4: -111.2198334, 85.4972687, -119.9568024, 91.9793015, -203.1991272, 205.4540710
5: -99.2353973, 77.3344879, -107.0011215, 83.3186722, -182.5540314, 184.3356018
6: -95.5971298, 91.3086395, -103.0942459, 98.3950500, -193.9921265, 194.4028778
7: -104.2331772, 87.4791412, -112.2794952, 94.2552719, -198.4884491, 199.7586365
8: -124.5192184, 85.1616516, -134.0392761, 91.4946136, -216.0138245, 219.2009277
9: -95.1064224, 93.3583527, -102.4540939, 100.5486450, -195.6550598, 195.8124390

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

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
type: B, layer: 1, pos: 204

## Relational analysis of IS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 133

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3195572, upper bound: 197.3195102
time: 6.99 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3192993, upper bound: 197.3192993
time: 7.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.17 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 27.17
Output dim: 4, lower bound: -197.3253311, upper bound: 197.3256712
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 27.17
Output dim: 4, lower bound: -197.3249350, upper bound: 197.3252364
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 27.17
Output dim: 4, lower bound: -197.3195572, upper bound: 197.3195102
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 27.17
Output dim: 4, lower bound: -197.3192993, upper bound: 197.3192993

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -102.9410248, 81.7447433, -89.5212708, 71.1000977, -174.0411224, 171.2660217
1: -85.8245621, 72.6093826, -74.5649796, 63.1741829, -148.9987488, 147.1743164
2: -113.2222519, 73.9574356, -98.4693985, 64.3496857, -177.5719299, 172.4268188
3: -120.5640182, 63.3619385, -104.8402557, 55.0911865, -175.6551971, 168.2021637
4: -110.7913055, 85.1991653, -96.4002228, 74.2435303, -185.0348053, 181.5993958
5: -98.8712540, 77.0543289, -85.9492569, 67.0725250, -165.9437866, 163.0035706
6: -95.2419357, 90.9865417, -82.9983368, 79.1548996, -174.3968048, 173.9848480
7: -103.8526840, 87.1619644, -90.4082642, 75.9232483, -179.7758789, 177.5702209
8: -124.1013565, 84.8834229, -108.0918121, 73.7984543, -197.8997955, 192.9752350
9: -94.7521133, 93.0418930, -82.5512772, 81.0500488, -175.8021545, 175.5931396

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 111

## Relational analysis of IS_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2871672, upper bound: 197.2880791
time: 10.10 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3224586, upper bound: 197.3230002
time: 11.44 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -103.1804886, 81.9344482, -102.6306992, 81.5010605, -184.6815338, 184.5651550
1: -86.0225372, 72.7778625, -85.5639343, 72.3918686, -158.4143982, 158.3417969
2: -113.4852219, 74.1267166, -112.8810959, 73.7361374, -187.2213440, 187.0078125
3: -120.8446350, 63.5091896, -120.2014618, 63.1723213, -184.0169525, 183.7106323
4: -111.0530319, 85.3952866, -110.4610596, 84.9466095, -195.9995880, 195.8563538
5: -99.1000443, 77.2317810, -98.5740051, 76.8282242, -175.9282532, 175.8057861
6: -95.4640045, 91.1977997, -94.9608078, 90.7123718, -186.1763763, 186.1585999
7: -104.0926666, 87.3629990, -103.5428925, 86.9052124, -190.9978790, 190.9058838
8: -124.3911591, 85.0781174, -123.7307816, 84.6244659, -209.0156250, 208.8088837
9: -94.9717178, 93.2561188, -94.4734116, 92.7677917, -187.7395020, 187.7295227

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 111

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3029843, upper bound: 197.3031026
time: 13.82 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3220564, upper bound: 197.3225175
time: 9.74 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -88.4576797, 70.2519913, -109.7536850, 87.1516266, -175.6092834, 180.0056763
1: -73.6527939, 62.4161148, -91.3395538, 77.3223190, -150.9751129, 153.7556763
2: -97.2861633, 63.5857506, -120.6735229, 78.7462997, -176.0324707, 184.2592773
3: -103.5907288, 54.4138947, -128.5340576, 67.3156128, -170.9063416, 182.9479523
4: -95.2585373, 73.3621979, -118.2121277, 90.6662598, -185.9247742, 191.5742798
5: -84.9269180, 66.2872314, -105.4594116, 82.1340790, -167.0610046, 171.7466431
6: -82.0217285, 78.2017746, -101.6144638, 96.9750366, -178.9967651, 179.8162384
7: -89.3375702, 75.0322876, -110.6669006, 92.9118652, -182.2494202, 185.6991882
8: -106.7792358, 72.8935471, -132.1097870, 90.1751709, -196.9544067, 205.0033264
9: -81.5856476, 80.0848465, -100.9883499, 99.1206894, -180.7063293, 181.0731812

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2809835, upper bound: 197.2806301
time: 7.54 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3165408, upper bound: 197.3164571
time: 5.79 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -101.5742874, 80.6587601, -109.9977264, 87.3449936, -188.9192505, 190.6564789
1: -84.6580658, 71.6390991, -91.5411682, 77.4940186, -162.1520844, 163.1802673
2: -111.7056427, 72.9779129, -120.9416809, 78.9186172, -190.6242676, 193.9195862
3: -118.9603424, 62.5002098, -128.8199768, 67.4655991, -186.4259338, 191.3201752
4: -109.3272018, 84.0708694, -118.4789963, 90.8662643, -200.1934662, 202.5498352
5: -97.5580826, 76.0485001, -105.6922989, 82.3150787, -179.8731689, 181.7407684
6: -93.9909363, 89.7657471, -101.8406448, 97.1905365, -191.1814423, 191.6063843
7: -102.4797974, 86.0200272, -110.9113235, 93.1168671, -195.5966492, 196.9313354
8: -122.4274979, 83.7261734, -132.4051361, 90.3733978, -212.8008423, 216.1313171
9: -93.5150070, 91.8093033, -101.2124023, 99.3390808, -192.8540955, 193.0216980

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.15 seconds

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
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 111

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2809833, upper bound: 197.2806301
time: 8.23 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3162730, upper bound: 197.3162730
time: 7.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 37.16 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.2871672, upper bound: 197.2880791
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.3224586, upper bound: 197.3230002
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.3029843, upper bound: 197.3031026
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.3220564, upper bound: 197.3225175
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.2809835, upper bound: 197.2806301
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.3165408, upper bound: 197.3164571
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.2809833, upper bound: 197.2806301
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 37.16
Output dim: 4, lower bound: -197.3162730, upper bound: 197.3162730

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -88.1211090, 70.0802841, -79.9475784, 63.5698357, -151.6909485, 150.0278625
1: -73.5217743, 62.2655716, -66.6148834, 56.4882278, -130.0099640, 128.8804321
2: -96.9565201, 63.4906769, -87.9569168, 57.5897179, -154.5462341, 151.4476013
3: -103.2592392, 54.2752457, -93.6583099, 49.2110977, -152.4703217, 147.9335632
4: -94.9275208, 73.1565170, -86.1523514, 66.4630051, -161.3905182, 159.3088684
5: -84.7212219, 66.1260681, -76.8092880, 60.0059090, -144.7271118, 142.9353638
6: -81.7328491, 77.9317856, -74.2830124, 70.7197800, -152.4526367, 152.2147980
7: -89.0132523, 74.8759842, -80.8198776, 67.9892883, -157.0025330, 155.6958466
8: -106.3297501, 72.7208023, -96.6194534, 65.9306564, -172.2604065, 169.3402405
9: -81.2484055, 79.6792908, -73.8160706, 72.4062958, -153.6546936, 153.4953613

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 111

## Relational analysis of IS_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2869278, upper bound: 197.2877230
time: 8.71 seconds

## Relational analysis of IS_B1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2869278, upper bound: 197.2880791
time: 9.90 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -95.7799683, 76.1089935, -83.9723587, 66.7309799, -162.5109406, 160.0812988
1: -79.8599396, 67.6070633, -69.9428024, 59.2989693, -139.1588745, 137.5498657
2: -105.3571930, 68.9039383, -92.3751907, 60.4331894, -165.7903595, 161.2791290
3: -112.2094421, 58.9730988, -98.3655701, 51.6907959, -163.9002380, 157.3386688
4: -103.1259689, 79.3616257, -90.4606094, 69.7221222, -172.8480530, 169.8222351
5: -92.0260925, 71.7691345, -80.6456146, 62.9778633, -155.0039368, 152.4147491
6: -88.7195892, 84.6693268, -77.9449844, 74.2605896, -162.9801788, 162.6143036
7: -96.6966324, 81.2192001, -84.8622894, 71.3202820, -168.0169067, 166.0814819
8: -115.5074692, 78.9903107, -101.4325180, 69.2306137, -184.7380829, 180.4228210
9: -88.2181625, 86.5775604, -77.4869843, 76.0402374, -164.2583466, 164.0645447

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_B1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3030067, upper bound: 197.3031448
time: 11.56 seconds

## Relational analysis of IS_B1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3030067, upper bound: 197.3230002
time: 11.54 seconds

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -93.3741913, 74.2177505, -87.8060989, 69.8329086, -163.2070923, 162.0238495
1: -77.8794174, 65.9299393, -73.2573853, 62.0446434, -139.9240417, 139.1873169
2: -102.7158508, 67.2002106, -96.6097260, 63.2656670, -165.9815216, 163.8099213
3: -109.3881226, 57.4877319, -102.8916779, 54.0826035, -163.4706879, 160.3793945
4: -100.5598145, 77.4245377, -94.5923615, 72.9002304, -173.4600525, 172.0169067
5: -89.7339706, 69.9935150, -84.4199142, 65.8969803, -155.6309509, 154.4134064
6: -86.5397339, 82.5568237, -81.4476013, 77.6534042, -164.1931458, 164.0044250
7: -94.2708969, 79.2338943, -88.6987000, 74.6156540, -168.8865509, 167.9325714
8: -112.6381836, 77.0195923, -105.9538040, 72.4581223, -185.0962372, 182.9733887
9: -86.0279007, 84.4000702, -80.9659729, 79.4012070, -165.4291077, 165.3660278

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_B2_B1_A1

### Relational analysis result of IS_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2869278, upper bound: 197.2877230
time: 9.82 seconds

## Relational analysis of IS_B1_B2_B1_A2

### Relational analysis result of IS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2869278, upper bound: 197.3031026
time: 9.89 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -97.7833557, 77.6873322, -95.4388885, 75.8414307, -173.6247864, 173.1262207
1: -81.5280228, 69.0078125, -79.5740280, 67.3682404, -148.8962708, 148.5818329
2: -107.5574112, 70.3178558, -104.9826584, 68.6610489, -176.2184601, 175.3005066
3: -114.5478439, 60.2020340, -111.8112183, 58.7648544, -173.3126831, 172.0132446
4: -105.2756042, 80.9958878, -102.7626801, 79.0842438, -184.3598328, 183.7585754
5: -93.9411469, 73.2485199, -91.6996002, 71.5206299, -165.4617767, 164.9480591
6: -90.5479126, 86.4368973, -88.4104004, 84.3683319, -174.9162292, 174.8472900
7: -98.6989975, 82.8837128, -96.3562927, 80.9370651, -179.6360626, 179.2399902
8: -117.9131241, 80.6374588, -115.1004257, 78.7061920, -196.6193085, 195.7378693
9: -90.0477524, 88.3838272, -87.9116821, 86.2763596, -176.3241119, 176.2954559

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B1_B2_B2_A1

### Relational analysis result of IS_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2871672, upper bound: 197.2880791
time: 9.20 seconds

## Relational analysis of IS_B1_B2_B2_A2

### Relational analysis result of IS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2871672, upper bound: 197.3225175
time: 9.00 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -78.8678589, 62.7088547, -95.0148010, 75.5496140, -154.4174652, 157.7236328
1: -65.6886673, 55.7184715, -79.0993347, 67.0297699, -132.7184448, 134.8177795
2: -86.7557449, 56.8141594, -104.4962997, 68.3334198, -155.0891724, 161.3104248
3: -92.3905106, 48.5235825, -111.3244400, 58.2730331, -150.6635132, 159.8480225
4: -84.9932861, 65.5685501, -102.4373474, 78.6882782, -163.6815338, 168.0058746
5: -75.7713165, 59.2089081, -91.3860474, 71.2617111, -147.0330200, 150.5949554
6: -73.2915497, 69.7523956, -88.1778336, 83.9919510, -157.2834930, 157.9302368
7: -79.7329865, 67.0850830, -95.9051208, 80.6865082, -160.4194794, 162.9902039
8: -95.2871857, 65.0119934, -114.4288483, 78.0803909, -173.3675842, 179.4408417
9: -72.8358765, 71.4264221, -87.5566711, 85.8298874, -158.6657410, 158.9830933

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2783778, upper bound: 197.2783797
time: 6.68 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2783778, upper bound: 197.2806301
time: 5.93 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -82.9143143, 65.8873138, -102.6263046, 81.5444412, -164.4587555, 168.5136108
1: -69.0352402, 58.5447769, -85.4046173, 72.3433075, -141.3785400, 143.9493866
2: -91.1981201, 59.6733208, -112.8468475, 73.7161484, -164.9142609, 172.5201416
3: -97.1225739, 51.0167770, -120.2175369, 62.9456940, -160.0682678, 171.2343140
4: -89.3247375, 68.8455658, -110.5831375, 84.8589783, -174.1837158, 179.4286957
5: -79.6286926, 62.1966782, -98.6465302, 76.8737946, -156.5024872, 160.8431854
6: -76.9735413, 73.3124390, -95.1219559, 90.6887894, -167.6623230, 168.4343719
7: -83.7973557, 70.4343414, -103.5442200, 86.9975510, -170.7948608, 173.9785614
8: -100.1263351, 68.3300781, -123.5547256, 84.3117371, -184.4380493, 191.8847961
9: -76.5264587, 75.0798569, -94.4835510, 92.6871490, -169.2136078, 169.5633850

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2801890, upper bound: 197.2804844
time: 6.68 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2801890, upper bound: 197.3164571
time: 6.92 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -91.7568588, 72.9330368, -95.2493820, 75.7353973, -167.4922485, 168.1823883
1: -76.5052261, 64.7829666, -79.2929459, 67.1946869, -143.6999207, 144.0758820
2: -100.9239273, 66.0432510, -104.7537231, 68.4986954, -169.4226227, 170.7969666
3: -107.4908600, 56.4714546, -111.5994415, 58.4172287, -165.9080505, 168.0708771
4: -98.8221130, 76.0910492, -102.6938705, 78.8806610, -177.7027740, 178.7848969
5: -88.1814728, 68.8020248, -91.6100922, 71.4357834, -159.6172485, 160.4121094
6: -85.0562210, 81.1150055, -88.3953857, 84.1989365, -169.2551422, 169.5103912
7: -92.6467285, 77.8816528, -96.1397247, 80.8836365, -173.5303345, 174.0213776
8: -110.6605148, 75.6579819, -114.7126160, 78.2711487, -188.9316406, 190.3706055
9: -84.5609207, 82.9430695, -87.7723541, 86.0399399, -170.6008606, 170.7154236

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2784058, upper bound: 197.2784058
time: 5.78 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2784058, upper bound: 197.2806302
time: 4.76 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -96.1807022, 76.4146729, -102.8470230, 81.7195663, -177.9002380, 179.2616882
1: -80.1668625, 67.8717041, -85.5870209, 72.4988098, -152.6656647, 153.4587250
2: -105.7820816, 69.1717529, -113.0896149, 73.8720093, -179.6540680, 182.2613525
3: -112.6679764, 59.1953316, -120.4764252, 63.0814934, -175.7494659, 179.6717224
4: -103.5537643, 79.6748047, -110.8248672, 85.0401230, -188.5938873, 190.4996643
5: -92.4029083, 72.0680008, -98.8571320, 77.0376282, -169.4405365, 170.9251099
6: -89.0781860, 85.0082245, -95.3268051, 90.8838348, -179.9620209, 180.3350220
7: -97.0899353, 81.5440063, -103.7654724, 87.1831741, -184.2731018, 185.3094330
8: -115.9542160, 79.2884369, -123.8222122, 84.4908905, -200.4450989, 203.1106262
9: -88.5941849, 86.9405441, -94.6864929, 92.8849030, -181.4790802, 181.6270142

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 105
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2806301, upper bound: 197.2809835
time: 6.90 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2806301, upper bound: 197.3162730
time: 7.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.00 seconds
IS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2869278, upper bound: 197.2877230
IS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2869278, upper bound: 197.2880791
IS_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.3030067, upper bound: 197.3031448
IS_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.3030067, upper bound: 197.3230002
IS_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2869278, upper bound: 197.2877230
IS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2869278, upper bound: 197.3031026
IS_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2871672, upper bound: 197.2880791
IS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2871672, upper bound: 197.3225175
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2783778, upper bound: 197.2783797
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2783778, upper bound: 197.2806301
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2801890, upper bound: 197.2804844
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2801890, upper bound: 197.3164571
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2784058, upper bound: 197.2784058
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2784058, upper bound: 197.2806302
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2806301, upper bound: 197.2809835
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 4, lower bound: -197.2806301, upper bound: 197.3162730

## BFS IS instance: IS_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -88.1211090, 70.0802841, -74.9886246, 59.6651535, -147.7862549, 145.0689087
1: -73.5217743, 62.2655716, -62.5009766, 53.0292816, -126.5510559, 124.7665482
2: -96.9565201, 63.4906769, -82.5195007, 54.0877686, -151.0442810, 146.0101776
3: -103.2592392, 54.2752457, -87.8738174, 46.1800003, -149.4392242, 142.1490631
4: -94.9275208, 73.1565170, -80.8369522, 62.4357529, -157.3632812, 153.9934692
5: -84.7212219, 66.1260681, -72.0786133, 56.3560486, -141.0772705, 138.2046814
6: -81.7328491, 77.9317856, -69.7453232, 66.3536224, -148.0864716, 147.6770935
7: -89.0132523, 74.8759842, -75.8559189, 63.8786163, -152.8918762, 150.7318726
8: -106.3297501, 72.7208023, -90.6654816, 61.8698006, -168.1995392, 163.3862762
9: -81.2484055, 79.6792908, -69.3041229, 67.9493866, -149.1977844, 148.9834137

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 105
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 111

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 111

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2528646, upper bound: 197.2527623
time: 9.52 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2456011, upper bound: 197.2457520
time: 10.95 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2584581, upper bound: 197.2588370
time: 8.22 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 108

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 233

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 108

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 168

## Relational analysis of IS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2867781, upper bound: 197.2875390
time: 8.80 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2868852, upper bound: 197.2876656
time: 9.03 seconds

## BFS IS instance: IS_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.1211090, 70.0802841, -82.1455536, 65.2926407, -153.4137421, 152.2258301
1: -73.5217743, 62.2655716, -68.4205017, 58.0232391, -131.5449982, 130.6860657
2: -96.9565201, 63.4906769, -90.3693314, 59.1443138, -156.1008301, 153.8600006
3: -103.2592392, 54.2752457, -96.2347412, 50.5704765, -153.8296967, 150.5099792
4: -94.9275208, 73.1565170, -88.5053024, 68.2332001, -163.1607056, 161.6618195
5: -84.7212219, 66.1260681, -78.8998489, 61.6296883, -146.3509064, 145.0259094
6: -81.7328491, 77.9317856, -76.2812271, 72.6497574, -154.3825989, 154.2129974
7: -89.0132523, 74.8759842, -83.0369797, 69.8059845, -158.8192444, 157.9129486
8: -106.3297501, 72.7208023, -99.2413940, 67.7255249, -174.0552673, 171.9621887
9: -81.2484055, 79.6792908, -75.8195267, 74.3911438, -155.6395416, 155.4988098

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of IS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=198.953369140625
rel_dist={4: [-197.4407374020123, 197.4407374020123]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1804.75 seconds
