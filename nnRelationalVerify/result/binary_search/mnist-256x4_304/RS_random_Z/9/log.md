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
execution time: IAR + LP analysis = 1.41 + 8.15 = 9.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4409435, upper bound: 197.4409435


# Binary Search by BASE starts (time budget: 1990.43 seconds, max iter: 100)

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
Binary search time: 32.84 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1957.59 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4269526, upper bound: 197.4269526
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4269526, upper bound: 197.4269526
time: 6.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.35
Output dim: 4, lower bound: -197.4269526, upper bound: 197.4269526
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.35
Output dim: 4, lower bound: -197.4269526, upper bound: 197.4269526

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4168224, upper bound: 197.4168224
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4168224, upper bound: 197.4168224
time: 5.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041854
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041854
time: 5.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 4, lower bound: -197.4168224, upper bound: 197.4168224
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 4, lower bound: -197.4168224, upper bound: 197.4168224
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041854
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.54
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041854

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4158982, upper bound: 197.4158978
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4158982, upper bound: 197.4158982
time: 5.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3469174
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3469174
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4013423, upper bound: 197.4013525
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4013526, upper bound: 197.4013422
time: 6.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041754
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041754, upper bound: 197.4041854
time: 5.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.4158982, upper bound: 197.4158978
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.4158982, upper bound: 197.4158982
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3469174
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3469174
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.4013423, upper bound: 197.4013525
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.4013526, upper bound: 197.4013422
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041754
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.70
Output dim: 4, lower bound: -197.4041754, upper bound: 197.4041854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4158400, upper bound: 197.4158404
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4158404, upper bound: 197.4158400
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4145091, upper bound: 197.4145138
time: 6.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4145132, upper bound: 197.4145091
time: 5.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3115951, upper bound: 197.3115951
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3115951, upper bound: 197.3115951
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3469139
time: 65.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3469139, upper bound: 197.3469174
time: 5.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3128490, upper bound: 197.3128485
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3128490, upper bound: 197.3128485
time: 4.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4013525, upper bound: 197.4013202
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4013352, upper bound: 197.4013423
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041754
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041847, upper bound: 197.4041754
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041754, upper bound: 197.4041744
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041580, upper bound: 197.4041854
time: 5.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4158400, upper bound: 197.4158404
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4158404, upper bound: 197.4158400
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4145091, upper bound: 197.4145138
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4145132, upper bound: 197.4145091
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.3115951, upper bound: 197.3115951
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.3115951, upper bound: 197.3115951
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3469139
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.3469139, upper bound: 197.3469174
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.3128490, upper bound: 197.3128485
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.3128490, upper bound: 197.3128485
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4013525, upper bound: 197.4013202
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4013352, upper bound: 197.4013423
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041754
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4041847, upper bound: 197.4041754
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4041754, upper bound: 197.4041744
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.94
Output dim: 4, lower bound: -197.4041580, upper bound: 197.4041854

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4132728, upper bound: 197.4132710
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4132712, upper bound: 197.4132728
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4114048, upper bound: 197.4114118
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4114003, upper bound: 197.4114138
time: 7.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3820536, upper bound: 197.3820535
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3820536, upper bound: 197.3820535
time: 5.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3399119, upper bound: 197.3399023
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3399119, upper bound: 197.3399023
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3115894, upper bound: 197.3115951
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3115951, upper bound: 197.3115894
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035590, upper bound: 197.3035431
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035431, upper bound: 197.3035590
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3468879
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3468881, upper bound: 197.3469139
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3469139, upper bound: 197.3469160
time: 18.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3469139, upper bound: 197.3469174
time: 6.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3128384, upper bound: 197.3128485
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3128490, upper bound: 197.3128366
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3031400, upper bound: 197.3031310
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3031323, upper bound: 197.3031377
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3966827, upper bound: 197.3966587
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3966689, upper bound: 197.3966768
time: 6.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3930230, upper bound: 197.3930943
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3930230, upper bound: 197.3930943
time: 8.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041577
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4041652, upper bound: 197.4041754
time: 6.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4021114, upper bound: 197.4020938
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4021117, upper bound: 197.4020853
time: 6.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3998477, upper bound: 197.3998512
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3998416, upper bound: 197.3998606
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4020770, upper bound: 197.4021186
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4020905, upper bound: 197.4021155
time: 12.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4132728, upper bound: 197.4132710
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4132712, upper bound: 197.4132728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4114048, upper bound: 197.4114118
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4114003, upper bound: 197.4114138
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3820536, upper bound: 197.3820535
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3820536, upper bound: 197.3820535
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3399119, upper bound: 197.3399023
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3399119, upper bound: 197.3399023
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3115894, upper bound: 197.3115951
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3115951, upper bound: 197.3115894
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3035590, upper bound: 197.3035431
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3035431, upper bound: 197.3035590
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3469174, upper bound: 197.3468879
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3468881, upper bound: 197.3469139
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3469139, upper bound: 197.3469160
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3469139, upper bound: 197.3469174
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3128384, upper bound: 197.3128485
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3128490, upper bound: 197.3128366
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3031400, upper bound: 197.3031310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3031323, upper bound: 197.3031377
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3966827, upper bound: 197.3966587
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3966689, upper bound: 197.3966768
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3930230, upper bound: 197.3930943
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3930230, upper bound: 197.3930943
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4041854, upper bound: 197.4041577
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4041652, upper bound: 197.4041754
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4021114, upper bound: 197.4020938
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4021117, upper bound: 197.4020853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3998477, upper bound: 197.3998512
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.3998416, upper bound: 197.3998606
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4020770, upper bound: 197.4021186
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.02
Output dim: 4, lower bound: -197.4020905, upper bound: 197.4021155

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4015742, upper bound: 197.4015506
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4015742, upper bound: 197.4015506
time: 6.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4123507, upper bound: 197.4123480
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4123483, upper bound: 197.4123500
time: 6.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4088167, upper bound: 197.4088220
time: 5.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4088163, upper bound: 197.4088230
time: 5.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4103913, upper bound: 197.4104020
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4103942, upper bound: 197.4103989
time: 6.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3789681, upper bound: 197.3789660
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3789681, upper bound: 197.3789659
time: 5.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.26 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=198.953369140625
rel_dist={4: [-197.44087218970873, 197.4408721892934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4309524, upper bound: 197.4309524
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4309524, upper bound: 197.4309524
time: 5.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.96
Output dim: 4, lower bound: -197.4309524, upper bound: 197.4309524
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.96
Output dim: 4, lower bound: -197.4309524, upper bound: 197.4309524

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3749355, upper bound: 197.3749355
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3749355, upper bound: 197.3749355
time: 6.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4158194, upper bound: 197.4158194
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4158194, upper bound: 197.4158194
time: 7.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 4, lower bound: -197.3749355, upper bound: 197.3749355
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 4, lower bound: -197.3749355, upper bound: 197.3749355
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 4, lower bound: -197.4158194, upper bound: 197.4158194
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 4, lower bound: -197.4158194, upper bound: 197.4158194

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3749354, upper bound: 197.3749355
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3749355, upper bound: 197.3749354
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050889, upper bound: 197.3050888
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050889, upper bound: 197.3050888
time: 6.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601257
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601257
time: 5.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4033886, upper bound: 197.4033886
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4033886, upper bound: 197.4033886
time: 7.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.3749354, upper bound: 197.3749355
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.3749355, upper bound: 197.3749354
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.3050889, upper bound: 197.3050888
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.3050889, upper bound: 197.3050888
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601257
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601257
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.4033886, upper bound: 197.4033886
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.97
Output dim: 4, lower bound: -197.4033886, upper bound: 197.4033886

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3521622, upper bound: 197.3521608
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3521622, upper bound: 197.3521608
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3524221, upper bound: 197.3524254
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3524221, upper bound: 197.3524254
time: 6.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050888, upper bound: 197.3050850
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050850, upper bound: 197.3050888
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050886, upper bound: 197.3050888
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050888, upper bound: 197.3050886
time: 5.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601232
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601232, upper bound: 197.3601257
time: 5.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601247
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601247, upper bound: 197.3601257
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3392193, upper bound: 197.3392193
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3392193, upper bound: 197.3392193
time: 5.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3903902, upper bound: 197.3903901
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3903902, upper bound: 197.3903901
time: 7.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3521622, upper bound: 197.3521608
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3521622, upper bound: 197.3521608
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3524221, upper bound: 197.3524254
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3524221, upper bound: 197.3524254
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3050888, upper bound: 197.3050850
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3050850, upper bound: 197.3050888
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3050886, upper bound: 197.3050888
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3050888, upper bound: 197.3050886
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601232
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3601232, upper bound: 197.3601257
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3601257, upper bound: 197.3601247
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3601247, upper bound: 197.3601257
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3392193, upper bound: 197.3392193
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3392193, upper bound: 197.3392193
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3903902, upper bound: 197.3903901
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.51
Output dim: 4, lower bound: -197.3903902, upper bound: 197.3903901

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3521622, upper bound: 197.3521568
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3521592, upper bound: 197.3521608
time: 6.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 113

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036952, upper bound: 197.3036948
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036952, upper bound: 197.3036948
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3066300, upper bound: 197.3066407
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3066300, upper bound: 197.3066407
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3459886, upper bound: 197.3459880
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3459885, upper bound: 197.3459883
time: 5.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050794, upper bound: 197.3050757
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050795, upper bound: 197.3050745
time: 5.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2871749, upper bound: 197.2871746
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2871747, upper bound: 197.2871753
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2995069, upper bound: 197.2995079
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2995069, upper bound: 197.2995079
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 223

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050888, upper bound: 197.3050855
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3050868, upper bound: 197.3050886
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3382796, upper bound: 197.3382774
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3382796, upper bound: 197.3382774
time: 6.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3027674, upper bound: 197.3027687
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3027674, upper bound: 197.3027687
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2996200, upper bound: 197.2996192
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2996200, upper bound: 197.2996192
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601247, upper bound: 197.3601224
time: 8.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3601232, upper bound: 197.3601257
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3278615, upper bound: 197.3278615
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3278615, upper bound: 197.3278615
time: 5.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3328603, upper bound: 197.3328534
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3328534, upper bound: 197.3328603
time: 6.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3777022, upper bound: 197.3776981
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3776981, upper bound: 197.3777022
time: 7.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3903889, upper bound: 197.3903901
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3903902, upper bound: 197.3903889
time: 6.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3521622, upper bound: 197.3521568
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3521592, upper bound: 197.3521608
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3036952, upper bound: 197.3036948
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3036952, upper bound: 197.3036948
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3066300, upper bound: 197.3066407
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3066300, upper bound: 197.3066407
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3459886, upper bound: 197.3459880
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3459885, upper bound: 197.3459883
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3050794, upper bound: 197.3050757
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3050795, upper bound: 197.3050745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.2871749, upper bound: 197.2871746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.2871747, upper bound: 197.2871753
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.2995069, upper bound: 197.2995079
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.2995069, upper bound: 197.2995079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3050888, upper bound: 197.3050855
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3050868, upper bound: 197.3050886
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3382796, upper bound: 197.3382774
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3382796, upper bound: 197.3382774
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3027674, upper bound: 197.3027687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3027674, upper bound: 197.3027687
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.2996200, upper bound: 197.2996192
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.2996200, upper bound: 197.2996192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3601247, upper bound: 197.3601224
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3601232, upper bound: 197.3601257
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3278615, upper bound: 197.3278615
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3278615, upper bound: 197.3278615
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3328603, upper bound: 197.3328534
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3328534, upper bound: 197.3328603
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3777022, upper bound: 197.3776981
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3776981, upper bound: 197.3777022
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3903889, upper bound: 197.3903901
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.64
Output dim: 4, lower bound: -197.3903902, upper bound: 197.3903889

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3463857, upper bound: 197.3463794
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3463857, upper bound: 197.3463794
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3521592, upper bound: 197.3521565
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3521538, upper bound: 197.3521608
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2989871, upper bound: 197.2989871
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2989870, upper bound: 197.2989873
time: 6.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 147

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2930251, upper bound: 197.2930274
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2930258, upper bound: 197.2930261
time: 5.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2679240, upper bound: 197.2679223
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2679240, upper bound: 197.2679223
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3028185, upper bound: 197.3028271
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3028200, upper bound: 197.3028231
time: 5.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.3463857, upper bound: 197.3463794
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.3463857, upper bound: 197.3463794
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.3521592, upper bound: 197.3521565
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.3521538, upper bound: 197.3521608
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.2989871, upper bound: 197.2989871
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.2989870, upper bound: 197.2989873
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.2930251, upper bound: 197.2930274
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.2930258, upper bound: 197.2930261
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.2679240, upper bound: 197.2679223
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.2679240, upper bound: 197.2679223
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.3028185, upper bound: 197.3028271
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.57
Output dim: 4, lower bound: -197.3028200, upper bound: 197.3028231
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3459886, upper bound: 197.3459880
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3459885, upper bound: 197.3459883
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3050794, upper bound: 197.3050757
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3050795, upper bound: 197.3050745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.2871749, upper bound: 197.2871746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.2871747, upper bound: 197.2871753
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.2995069, upper bound: 197.2995079
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.2995069, upper bound: 197.2995079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3050888, upper bound: 197.3050855
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3050868, upper bound: 197.3050886
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3382796, upper bound: 197.3382774
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3382796, upper bound: 197.3382774
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3027674, upper bound: 197.3027687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3027674, upper bound: 197.3027687
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.2996200, upper bound: 197.2996192
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.2996200, upper bound: 197.2996192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3601247, upper bound: 197.3601224
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3601232, upper bound: 197.3601257
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3278615, upper bound: 197.3278615
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3278615, upper bound: 197.3278615
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3328603, upper bound: 197.3328534
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3328534, upper bound: 197.3328603
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3777022, upper bound: 197.3776981
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3776981, upper bound: 197.3777022
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3903889, upper bound: 197.3903901
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.57
Output dim: 4, lower bound: -197.3903902, upper bound: 197.3903889
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=198.953369140625
rel_dist={4: [-197.44083159618555, 197.44083163160866]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401221, upper bound: 197.4401228
time: 8.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401228, upper bound: 197.4401221
time: 7.03 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.33
Output dim: 4, lower bound: -197.4401221, upper bound: 197.4401228
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.33
Output dim: 4, lower bound: -197.4401228, upper bound: 197.4401221

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 233

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4229021, upper bound: 197.4229021
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4229021, upper bound: 197.4229021
time: 7.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4375344, upper bound: 197.4375303
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4375321, upper bound: 197.4375323
time: 8.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.76
Output dim: 4, lower bound: -197.4229021, upper bound: 197.4229021
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.76
Output dim: 4, lower bound: -197.4229021, upper bound: 197.4229021
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.76
Output dim: 4, lower bound: -197.4375344, upper bound: 197.4375303
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.76
Output dim: 4, lower bound: -197.4375321, upper bound: 197.4375323

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182431, upper bound: 197.4182431
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182431, upper bound: 197.4182431
time: 7.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4228997, upper bound: 197.4229021
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4229021, upper bound: 197.4229005
time: 9.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4262893, upper bound: 197.4262870
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4262870, upper bound: 197.4262878
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3736871, upper bound: 197.3736842
time: 7.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3736871, upper bound: 197.3736842
time: 7.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.40 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4182431, upper bound: 197.4182431
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4182431, upper bound: 197.4182431
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4228997, upper bound: 197.4229021
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4229021, upper bound: 197.4229005
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4262893, upper bound: 197.4262870
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.4262870, upper bound: 197.4262878
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.3736871, upper bound: 197.3736842
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.40
Output dim: 4, lower bound: -197.3736871, upper bound: 197.3736842

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182431, upper bound: 197.4182423
time: 8.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182421, upper bound: 197.4182431
time: 8.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3390919, upper bound: 197.3390918
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3390919, upper bound: 197.3390918
time: 5.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182431
time: 7.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182431
time: 9.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3470064, upper bound: 197.3470051
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3470064, upper bound: 197.3470051
time: 7.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4262893, upper bound: 197.4262847
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4262861, upper bound: 197.4262870
time: 8.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3464319, upper bound: 197.3464321
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3464319, upper bound: 197.3464321
time: 7.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3686790, upper bound: 197.3686753
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3686787, upper bound: 197.3686753
time: 6.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3647530, upper bound: 197.3647530
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3647530, upper bound: 197.3647530
time: 6.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 15.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.4182431, upper bound: 197.4182423
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.4182421, upper bound: 197.4182431
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3390919, upper bound: 197.3390918
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3390919, upper bound: 197.3390918
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182431
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182431
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3470064, upper bound: 197.3470051
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3470064, upper bound: 197.3470051
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.4262893, upper bound: 197.4262847
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.4262861, upper bound: 197.4262870
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3464319, upper bound: 197.3464321
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3464319, upper bound: 197.3464321
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3686790, upper bound: 197.3686753
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3686787, upper bound: 197.3686753
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3647530, upper bound: 197.3647530
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 15.32
Output dim: 4, lower bound: -197.3647530, upper bound: 197.3647530

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3804460, upper bound: 197.3804454
time: 7.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3804460, upper bound: 197.3804454
time: 8.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3848941, upper bound: 197.3848947
time: 8.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3848941, upper bound: 197.3848947
time: 8.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 133

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3275107, upper bound: 197.3275114
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3275107, upper bound: 197.3275114
time: 6.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3165250, upper bound: 197.3165247
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3165250, upper bound: 197.3165247
time: 6.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3123828, upper bound: 197.3123840
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3123828, upper bound: 197.3123840
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182423
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182431
time: 7.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 54

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3465601, upper bound: 197.3465588
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3465603, upper bound: 197.3465584
time: 6.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3470064, upper bound: 197.3470045
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3470056, upper bound: 197.3470051
time: 6.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4147991, upper bound: 197.4147978
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4147991, upper bound: 197.4147978
time: 6.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4262861, upper bound: 197.4262802
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4262797, upper bound: 197.4262870
time: 6.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 226

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3325076, upper bound: 197.3325066
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3325076, upper bound: 197.3325066
time: 5.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3114024, upper bound: 197.3114033
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3114024, upper bound: 197.3114033
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3686772, upper bound: 197.3686736
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3686768, upper bound: 197.3686736
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3636431, upper bound: 197.3636436
time: 7.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3636431, upper bound: 197.3636436
time: 7.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 72

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3476194, upper bound: 197.3476191
time: 8.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3476194, upper bound: 197.3476191
time: 8.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3382969, upper bound: 197.3382977
time: 8.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3382969, upper bound: 197.3382977
time: 9.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 19.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3804460, upper bound: 197.3804454
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3804460, upper bound: 197.3804454
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3848941, upper bound: 197.3848947
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3848941, upper bound: 197.3848947
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3275107, upper bound: 197.3275114
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3275107, upper bound: 197.3275114
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3165250, upper bound: 197.3165247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3165250, upper bound: 197.3165247
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3123828, upper bound: 197.3123840
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3123828, upper bound: 197.3123840
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182423
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3465601, upper bound: 197.3465588
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3465603, upper bound: 197.3465584
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3470064, upper bound: 197.3470045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3470056, upper bound: 197.3470051
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.4147991, upper bound: 197.4147978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.4147991, upper bound: 197.4147978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.4262861, upper bound: 197.4262802
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.4262797, upper bound: 197.4262870
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3325076, upper bound: 197.3325066
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3325076, upper bound: 197.3325066
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3114024, upper bound: 197.3114033
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3114024, upper bound: 197.3114033
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3686772, upper bound: 197.3686736
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3686768, upper bound: 197.3686736
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3636431, upper bound: 197.3636436
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3636431, upper bound: 197.3636436
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3476194, upper bound: 197.3476191
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3476194, upper bound: 197.3476191
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3382969, upper bound: 197.3382977
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 19.63
Output dim: 4, lower bound: -197.3382969, upper bound: 197.3382977

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3657156, upper bound: 197.3657151
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3657156, upper bound: 197.3657151
time: 7.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3498439, upper bound: 197.3498439
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3498439, upper bound: 197.3498439
time: 6.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 111

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3848962, upper bound: 197.3848897
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3848899, upper bound: 197.3848947
time: 7.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3848961, upper bound: 197.3848947
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3848962, upper bound: 197.3848942
time: 6.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3158027, upper bound: 197.3158012
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3158011, upper bound: 197.3158028
time: 6.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 68

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3275011, upper bound: 197.3274989
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3274984, upper bound: 197.3275018
time: 7.32 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3657156, upper bound: 197.3657151
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3657156, upper bound: 197.3657151
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3498439, upper bound: 197.3498439
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3498439, upper bound: 197.3498439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3848962, upper bound: 197.3848897
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3848899, upper bound: 197.3848947
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3848961, upper bound: 197.3848947
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3848962, upper bound: 197.3848942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3158027, upper bound: 197.3158012
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3158011, upper bound: 197.3158028
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3275011, upper bound: 197.3274989
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.44
Output dim: 4, lower bound: -197.3274984, upper bound: 197.3275018
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3165250, upper bound: 197.3165247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3165250, upper bound: 197.3165247
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3123828, upper bound: 197.3123840
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3123828, upper bound: 197.3123840
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182423
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.4182407, upper bound: 197.4182431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3465601, upper bound: 197.3465588
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3465603, upper bound: 197.3465584
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3470064, upper bound: 197.3470045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3470056, upper bound: 197.3470051
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.4147991, upper bound: 197.4147978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.4147991, upper bound: 197.4147978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.4262861, upper bound: 197.4262802
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.4262797, upper bound: 197.4262870
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3325076, upper bound: 197.3325066
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3325076, upper bound: 197.3325066
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3114024, upper bound: 197.3114033
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3114024, upper bound: 197.3114033
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3686772, upper bound: 197.3686736
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3686768, upper bound: 197.3686736
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3636431, upper bound: 197.3636436
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3636431, upper bound: 197.3636436
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3476194, upper bound: 197.3476191
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3476194, upper bound: 197.3476191
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3382969, upper bound: 197.3382977
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.44
Output dim: 4, lower bound: -197.3382969, upper bound: 197.3382977
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=198.953369140625
rel_dist={4: [-197.4407374020123, 197.4407374020123]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1817.46 seconds
