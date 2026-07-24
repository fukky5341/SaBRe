## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 10.310653145
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540)
1: (-4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358)
2: (-5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447)
3: (-6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520)
4: (-6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002)
5: (-5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936)
6: (-4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792)
7: (-5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823)
8: (-7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416)
9: (-4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 5.26 = 6.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -10.8533233, upper bound: 10.8533231


# Binary Search by BASE starts (time budget: 1993.50 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=12.009641647338867
rel_dist={8: [-10.853321756160986, 10.853321760348969]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=12.009641647338867
rel_dist={8: [-10.85331894689149, 10.853318913028158]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=12.009641647338867
rel_dist={8: [-10.853314892972527, 10.853314942404314]}

## Binary Search Result
Binary search time: 20.73 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1972.78 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8523281, upper bound: 10.8523272
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8523275, upper bound: 10.8523273
time: 3.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.58
Output dim: 8, lower bound: -10.8523281, upper bound: 10.8523272
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.58
Output dim: 8, lower bound: -10.8523275, upper bound: 10.8523273

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500359
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
time: 2.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
time: 2.17 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500359
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495038, upper bound: 10.8495046
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495038, upper bound: 10.8495046
time: 2.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467808, upper bound: 10.8467817
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467808, upper bound: 10.8467817
time: 1.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487397, upper bound: 10.8487408
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487409, upper bound: 10.8487397
time: 2.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
time: 18.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8495038, upper bound: 10.8495046
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8495038, upper bound: 10.8495046
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8467808, upper bound: 10.8467817
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8467808, upper bound: 10.8467817
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8487397, upper bound: 10.8487408
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8487409, upper bound: 10.8487397
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.75
Output dim: 8, lower bound: -10.8500360, upper bound: 10.8500360

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3022051, upper bound: 10.3022051
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3022051, upper bound: 10.3022051
time: 1.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495046, upper bound: 10.8495046
time: 3.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495045, upper bound: 10.8495046
time: 2.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4169660, upper bound: 10.4169659
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4169660, upper bound: 10.4169659
time: 1.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448428, upper bound: 10.8448425
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8448428, upper bound: 10.8448425
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487397, upper bound: 10.8487408
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8487397, upper bound: 10.8487408
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8462660, upper bound: 10.8462499
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8462660, upper bound: 10.8462499
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500359, upper bound: 10.8500360
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8500359, upper bound: 10.8500360
time: 3.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8491306, upper bound: 10.8491306
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8491306, upper bound: 10.8491308
time: 2.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.3022051, upper bound: 10.3022051
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.3022051, upper bound: 10.3022051
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8495046, upper bound: 10.8495046
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8495045, upper bound: 10.8495046
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.4169660, upper bound: 10.4169659
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.4169660, upper bound: 10.4169659
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8448428, upper bound: 10.8448425
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8448428, upper bound: 10.8448425
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8487397, upper bound: 10.8487408
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8487397, upper bound: 10.8487408
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8462660, upper bound: 10.8462499
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8462660, upper bound: 10.8462499
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8500359, upper bound: 10.8500360
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8500359, upper bound: 10.8500360
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8491306, upper bound: 10.8491306
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.07
Output dim: 8, lower bound: -10.8491306, upper bound: 10.8491308

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495047, upper bound: 10.8495046
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8495047, upper bound: 10.8495045
time: 2.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484478
time: 2.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484481
time: 3.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1601630, upper bound: 10.1601630
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1601630, upper bound: 10.1601630
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1707426, upper bound: 10.1707427
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1707426, upper bound: 10.1707426
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8480575, upper bound: 10.8480573
time: 3.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8480575, upper bound: 10.8480573
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478006, upper bound: 10.8478020
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8478006, upper bound: 10.8478020
time: 2.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7086429, upper bound: 10.7086426
time: 1.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7086429, upper bound: 10.7086426
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8477206, upper bound: 10.8477218
time: 2.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8477206, upper bound: 10.8477217
time: 2.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8492714, upper bound: 10.8492709
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8492714, upper bound: 10.8492705
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460167, upper bound: 10.8460168
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460168, upper bound: 10.8460167
time: 2.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484481
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484480
time: 2.69 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.22 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8495047, upper bound: 10.8495046
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8495047, upper bound: 10.8495045
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484478
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484481
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.1601630, upper bound: 10.1601630
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.1601630, upper bound: 10.1601630
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.1707426, upper bound: 10.1707427
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.1707426, upper bound: 10.1707426
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8480575, upper bound: 10.8480573
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8480575, upper bound: 10.8480573
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8478006, upper bound: 10.8478020
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8478006, upper bound: 10.8478020
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7086429, upper bound: 10.7086426
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.7086429, upper bound: 10.7086426
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8477206, upper bound: 10.8477218
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8477206, upper bound: 10.8477217
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8492714, upper bound: 10.8492709
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8492714, upper bound: 10.8492705
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8460167, upper bound: 10.8460168
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8460168, upper bound: 10.8460167
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484481
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.22
Output dim: 8, lower bound: -10.8484477, upper bound: 10.8484480

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4284042, upper bound: 10.4284042
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4284042, upper bound: 10.4284042
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8047949, upper bound: 10.8047942
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8047949, upper bound: 10.8047942
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461149, upper bound: 10.8461113
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461159, upper bound: 10.8461092
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461149, upper bound: 10.8461113
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8461159, upper bound: 10.8461092
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6371278, upper bound: 10.6371277
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6371278, upper bound: 10.6371277
time: 1.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503901
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
time: 1.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8246388, upper bound: 10.8246388
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8246388, upper bound: 10.8246388
time: 9.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
time: 1.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8023221, upper bound: 10.8023227
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8023221, upper bound: 10.8023240
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8133657, upper bound: 10.8133657
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8133657, upper bound: 10.8133657
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
time: 1.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
time: 1.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6284916, upper bound: 10.6284916
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6284916, upper bound: 10.6284916
time: 2.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8449425, upper bound: 10.8449520
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8449456, upper bound: 10.8449466
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8471650, upper bound: 10.8471658
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8471650, upper bound: 10.8471658
time: 2.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460075, upper bound: 10.8460072
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460075, upper bound: 10.8460072
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475719, upper bound: 10.8475714
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475729, upper bound: 10.8475709
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7482871, upper bound: 10.7482871
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7482871, upper bound: 10.7482871
time: 1.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6873035, upper bound: 10.6873034
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6873035, upper bound: 10.6873034
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8443494, upper bound: 10.8443483
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8443494, upper bound: 10.8443483
time: 2.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475722, upper bound: 10.8475718
time: 3.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8475722, upper bound: 10.8475720
time: 2.52 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.05 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.4284042, upper bound: 10.4284042
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.4284042, upper bound: 10.4284042
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8047949, upper bound: 10.8047942
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8047949, upper bound: 10.8047942
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8461149, upper bound: 10.8461113
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8461159, upper bound: 10.8461092
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8461149, upper bound: 10.8461113
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8461159, upper bound: 10.8461092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7600234, upper bound: 10.7600227
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6371278, upper bound: 10.6371277
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6371278, upper bound: 10.6371277
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503901
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.3503901, upper bound: 10.3503902
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8246388, upper bound: 10.8246388
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8246388, upper bound: 10.8246388
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8023221, upper bound: 10.8023227
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8023221, upper bound: 10.8023240
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8133657, upper bound: 10.8133657
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8133657, upper bound: 10.8133657
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6284916, upper bound: 10.6284916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6284916, upper bound: 10.6284916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8449425, upper bound: 10.8449520
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8449456, upper bound: 10.8449466
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8471650, upper bound: 10.8471658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8471650, upper bound: 10.8471658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8460075, upper bound: 10.8460072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8460075, upper bound: 10.8460072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8475719, upper bound: 10.8475714
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8475729, upper bound: 10.8475709
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7482871, upper bound: 10.7482871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.7482871, upper bound: 10.7482871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6873035, upper bound: 10.6873034
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.6873035, upper bound: 10.6873034
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8443494, upper bound: 10.8443483
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8443494, upper bound: 10.8443483
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8475722, upper bound: 10.8475718
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.05
Output dim: 8, lower bound: -10.8475722, upper bound: 10.8475720

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4023878, upper bound: 10.4023878
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4023878, upper bound: 10.4023878
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7534114, upper bound: 10.7534114
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7534114, upper bound: 10.7534114
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453561, upper bound: 10.8453548
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8453561, upper bound: 10.8453548
time: 2.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
time: 1.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450382, upper bound: 10.8450311
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8450373, upper bound: 10.8450320
time: 2.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5668620, upper bound: 10.5668621
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5668620, upper bound: 10.5668621
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7242316, upper bound: 10.7242316
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7242316, upper bound: 10.7242316
time: 2.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2816962, upper bound: 10.2816966
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2816962, upper bound: 10.2816966
time: 1.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5643666, upper bound: 10.5643685
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5643666, upper bound: 10.5643677
time: 1.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3161121, upper bound: 10.3161100
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3161113, upper bound: 10.3161112
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3161113, upper bound: 10.3161119
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3161076, upper bound: 10.3161121
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3161097, upper bound: 10.3161119
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3161082, upper bound: 10.3161121
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2468147, upper bound: 10.2468147
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2468147, upper bound: 10.2468147
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
time: 2.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
time: 1.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
time: 1.53 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.87 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3478280, upper bound: 10.3478280
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4023878, upper bound: 10.4023878
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4023878, upper bound: 10.4023878
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.7534114, upper bound: 10.7534114
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.7534114, upper bound: 10.7534114
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.8453561, upper bound: 10.8453548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.8453561, upper bound: 10.8453548
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.8450382, upper bound: 10.8450311
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.8450373, upper bound: 10.8450320
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5668620, upper bound: 10.5668621
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5668620, upper bound: 10.5668621
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.7242316, upper bound: 10.7242316
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.7242316, upper bound: 10.7242316
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.2816962, upper bound: 10.2816966
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.2816962, upper bound: 10.2816966
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5643666, upper bound: 10.5643685
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5643666, upper bound: 10.5643677
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3161121, upper bound: 10.3161100
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3161113, upper bound: 10.3161112
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3161113, upper bound: 10.3161119
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3161076, upper bound: 10.3161121
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3161097, upper bound: 10.3161119
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.3161082, upper bound: 10.3161121
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.2468147, upper bound: 10.2468147
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.2468147, upper bound: 10.2468147
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4530537, upper bound: 10.4530546
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.5421506, upper bound: 10.5421506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.7061123, upper bound: 10.7061123
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 8, lower bound: -10.4180457, upper bound: 10.4180457
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8133657, upper bound: 10.8133657
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8133657, upper bound: 10.8133657
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6487572, upper bound: 10.6487572
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.7086426, upper bound: 10.7086427
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6284916, upper bound: 10.6284916
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6284916, upper bound: 10.6284916
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8449425, upper bound: 10.8449520
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8449456, upper bound: 10.8449466
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8471650, upper bound: 10.8471658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8471650, upper bound: 10.8471658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8460075, upper bound: 10.8460072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8460075, upper bound: 10.8460072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8475719, upper bound: 10.8475714
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8475729, upper bound: 10.8475709
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.7482871, upper bound: 10.7482871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.7482871, upper bound: 10.7482871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6873035, upper bound: 10.6873034
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.6873035, upper bound: 10.6873034
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8443494, upper bound: 10.8443483
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8443494, upper bound: 10.8443483
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8475722, upper bound: 10.8475718
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 8, lower bound: -10.8475722, upper bound: 10.8475720
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=12.009641647338867
rel_dist={8: [-10.853321756160986, 10.853321760348969]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8528000, upper bound: 10.8528002
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8528000, upper bound: 10.8528001
time: 7.22 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.69
Output dim: 8, lower bound: -10.8528000, upper bound: 10.8528002
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.69
Output dim: 8, lower bound: -10.8528000, upper bound: 10.8528001

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484270, upper bound: 10.8484270
time: 2.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8484270, upper bound: 10.8484266
time: 5.20 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8516279, upper bound: 10.8516281
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8516283, upper bound: 10.8516280
time: 3.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.03 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 8, lower bound: -10.8484270, upper bound: 10.8484270
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 8, lower bound: -10.8484270, upper bound: 10.8484266
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 8, lower bound: -10.8516279, upper bound: 10.8516281
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.03
Output dim: 8, lower bound: -10.8516283, upper bound: 10.8516280

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476041, upper bound: 10.8476041
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476041, upper bound: 10.8476041
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5066066, upper bound: 10.5066066
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5066066, upper bound: 10.5066066
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8496298, upper bound: 10.8496299
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8496298, upper bound: 10.8496299
time: 2.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476574, upper bound: 10.8476577
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476574, upper bound: 10.8476577
time: 3.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.8476041, upper bound: 10.8476041
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.8476041, upper bound: 10.8476041
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.5066066, upper bound: 10.5066066
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.5066066, upper bound: 10.5066066
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.8496298, upper bound: 10.8496299
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.8496298, upper bound: 10.8496299
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.8476574, upper bound: 10.8476577
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.09
Output dim: 8, lower bound: -10.8476574, upper bound: 10.8476577

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
time: 5.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8436305, upper bound: 10.8436314
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8436314, upper bound: 10.8436305
time: 3.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
time: 17.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
time: 1.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
time: 13.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
time: 1.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8472482, upper bound: 10.8472482
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8472482, upper bound: 10.8472483
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8471701, upper bound: 10.8471701
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8471701, upper bound: 10.8471701
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 1.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8436305, upper bound: 10.8436314
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8436314, upper bound: 10.8436305
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8472482, upper bound: 10.8472482
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8472482, upper bound: 10.8472483
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8471701, upper bound: 10.8471701
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.8471701, upper bound: 10.8471701
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.73
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454094, upper bound: 10.8454096
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454096, upper bound: 10.8454094
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
time: 1.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8436261, upper bound: 10.8436271
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8436261, upper bound: 10.8436271
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
time: 1.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1773025, upper bound: 10.1773025
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1773025, upper bound: 10.1773025
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454416, upper bound: 10.8454413
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454416, upper bound: 10.8454412
time: 6.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8057470, upper bound: 10.8057460
time: 2.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8057476, upper bound: 10.8057459
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 2.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 1.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460177, upper bound: 10.8460185
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8460185, upper bound: 10.8460177
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 2.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8454094, upper bound: 10.8454096
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8454096, upper bound: 10.8454094
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8469588, upper bound: 10.8469588
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8436261, upper bound: 10.8436271
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8436261, upper bound: 10.8436271
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.2520292, upper bound: 10.2520292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.4548860, upper bound: 10.4548860
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.1773025, upper bound: 10.1773025
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.1773025, upper bound: 10.1773025
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8454416, upper bound: 10.8454413
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8454416, upper bound: 10.8454412
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8057470, upper bound: 10.8057460
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8057476, upper bound: 10.8057459
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8460177, upper bound: 10.8460185
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.8460185, upper bound: 10.8460177
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7380926, upper bound: 10.7380929
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7380926, upper bound: 10.7380929
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7586383, upper bound: 10.7586371
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7586381, upper bound: 10.7586375
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7067161, upper bound: 10.7067161
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7067161, upper bound: 10.7067161
time: 3.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4430430, upper bound: 10.4430430
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4430430, upper bound: 10.4430430
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7209803, upper bound: 10.7209806
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7209807, upper bound: 10.7209803
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6443688, upper bound: 10.6443688
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6443688, upper bound: 10.6443688
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
time: 2.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5270991, upper bound: 10.5270974
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5270991, upper bound: 10.5270974
time: 2.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2075791, upper bound: 10.2075791
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2075791, upper bound: 10.2075791
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3825829, upper bound: 10.3825822
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3825829, upper bound: 10.3825822
time: 3.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454416, upper bound: 10.8454409
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454411, upper bound: 10.8454412
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5534638, upper bound: 10.5534614
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5534638, upper bound: 10.5534614
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5215689, upper bound: 10.5215689
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5215689, upper bound: 10.5215689
time: 2.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7767334, upper bound: 10.7767307
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7767334, upper bound: 10.7767307
time: 1.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4982627, upper bound: 10.4982625
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4982627, upper bound: 10.4982625
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
time: 2.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
time: 2.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
time: 2.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7820302, upper bound: 10.7820300
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7820302, upper bound: 10.7820300
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8207900, upper bound: 10.8207901
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8207900, upper bound: 10.8207901
time: 2.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7294619, upper bound: 10.7294629
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7294619, upper bound: 10.7294629
time: 1.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2113845, upper bound: 10.2113845
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2113845, upper bound: 10.2113845
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2471465, upper bound: 10.2471465
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2471465, upper bound: 10.2471465
time: 1.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3399329, upper bound: 10.3399329
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3399329, upper bound: 10.3399329
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
time: 1.43 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 10.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7380926, upper bound: 10.7380929
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7380926, upper bound: 10.7380929
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7586383, upper bound: 10.7586371
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7586381, upper bound: 10.7586375
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7067161, upper bound: 10.7067161
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7067161, upper bound: 10.7067161
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.4430430, upper bound: 10.4430430
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.4430430, upper bound: 10.4430430
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7209803, upper bound: 10.7209806
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7209807, upper bound: 10.7209803
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.6443688, upper bound: 10.6443688
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.6443688, upper bound: 10.6443688
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5587353, upper bound: 10.5587355
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5270991, upper bound: 10.5270974
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5270991, upper bound: 10.5270974
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.2075791, upper bound: 10.2075791
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.2075791, upper bound: 10.2075791
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.3825829, upper bound: 10.3825822
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.3825829, upper bound: 10.3825822
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.8454416, upper bound: 10.8454409
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.8454411, upper bound: 10.8454412
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5534638, upper bound: 10.5534614
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5534638, upper bound: 10.5534614
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5215689, upper bound: 10.5215689
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.5215689, upper bound: 10.5215689
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7767334, upper bound: 10.7767307
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7767334, upper bound: 10.7767307
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.4982627, upper bound: 10.4982625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.4982627, upper bound: 10.4982625
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7841396, upper bound: 10.7841396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.6817387, upper bound: 10.6817387
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7892190, upper bound: 10.7892190
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7820302, upper bound: 10.7820300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7820302, upper bound: 10.7820300
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.8207900, upper bound: 10.8207901
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.8207900, upper bound: 10.8207901
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7294619, upper bound: 10.7294629
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.7294619, upper bound: 10.7294629
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.2113845, upper bound: 10.2113845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.2113845, upper bound: 10.2113845
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.2471465, upper bound: 10.2471465
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.2471465, upper bound: 10.2471465
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.3399329, upper bound: 10.3399329
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.3399329, upper bound: 10.3399329
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.55
Output dim: 8, lower bound: -10.4103129, upper bound: 10.4103129

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7042643, upper bound: 10.7042647
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7042643, upper bound: 10.7042647
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4612433, upper bound: 10.4612433
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4612433, upper bound: 10.4612433
time: 2.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4960977, upper bound: 10.4960977
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4960977, upper bound: 10.4960977
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5534751, upper bound: 10.5534751
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5534751, upper bound: 10.5534751
time: 2.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2336254, upper bound: 10.2336254
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2336254, upper bound: 10.2336254
time: 1.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7067161, upper bound: 10.7067161
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7067161, upper bound: 10.7067161
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4041231, upper bound: 10.4041231
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4041231, upper bound: 10.4041231
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2932083, upper bound: 10.2932083
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2932084, upper bound: 10.2932083
time: 1.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6707674, upper bound: 10.6707683
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6707674, upper bound: 10.6707683
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7209794, upper bound: 10.7209803
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7209807, upper bound: 10.7209793
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5778752, upper bound: 10.5778749
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5778749, upper bound: 10.5778752
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5936037, upper bound: 10.5936040
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5936037, upper bound: 10.5936040
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=12.009641647338867
rel_dist={8: [-10.85331894689149, 10.853318913028158]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8522109, upper bound: 10.8522114
time: 7.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8522117, upper bound: 10.8522112
time: 4.11 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.94
Output dim: 8, lower bound: -10.8522109, upper bound: 10.8522114
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.94
Output dim: 8, lower bound: -10.8522117, upper bound: 10.8522112

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476876, upper bound: 10.8476876
time: 9.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8476876, upper bound: 10.8476876
time: 8.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8522118, upper bound: 10.8522117
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8522113, upper bound: 10.8522112
time: 3.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 9.21 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.21
Output dim: 8, lower bound: -10.8476876, upper bound: 10.8476876
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.21
Output dim: 8, lower bound: -10.8476876, upper bound: 10.8476876
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 9.21
Output dim: 8, lower bound: -10.8522118, upper bound: 10.8522117
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 9.21
Output dim: 8, lower bound: -10.8522113, upper bound: 10.8522112

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7199170, upper bound: 10.7199170
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7199170, upper bound: 10.7199170
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467565
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467564
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8471931, upper bound: 10.8471931
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8471931, upper bound: 10.8471931
time: 3.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8504187, upper bound: 10.8504191
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8504188, upper bound: 10.8504191
time: 6.40 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.7199170, upper bound: 10.7199170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.7199170, upper bound: 10.7199170
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467565
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467564
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.8471931, upper bound: 10.8471931
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.8471931, upper bound: 10.8471931
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.8504187, upper bound: 10.8504191
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.94
Output dim: 8, lower bound: -10.8504188, upper bound: 10.8504191

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3788379, upper bound: 10.3788379
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3788379, upper bound: 10.3788379
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6713441, upper bound: 10.6713440
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6713440, upper bound: 10.6713441
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8458235, upper bound: 10.8458240
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8458235, upper bound: 10.8458240
time: 1.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467565
time: 2.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467564
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8465509, upper bound: 10.8465509
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8465509, upper bound: 10.8465509
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6941217, upper bound: 10.6941217
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6941217, upper bound: 10.6941217
time: 2.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 214

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8498778, upper bound: 10.8498778
time: 3.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8498778, upper bound: 10.8498779
time: 2.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8225366, upper bound: 10.8225368
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8225366, upper bound: 10.8225368
time: 2.09 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.3788379, upper bound: 10.3788379
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.3788379, upper bound: 10.3788379
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.6713441, upper bound: 10.6713440
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.6713440, upper bound: 10.6713441
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8458235, upper bound: 10.8458240
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8458235, upper bound: 10.8458240
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467565
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8467565, upper bound: 10.8467564
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8465509, upper bound: 10.8465509
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8465509, upper bound: 10.8465509
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.6941217, upper bound: 10.6941217
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.6941217, upper bound: 10.6941217
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8498778, upper bound: 10.8498778
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8498778, upper bound: 10.8498779
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8225366, upper bound: 10.8225368
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.46
Output dim: 8, lower bound: -10.8225366, upper bound: 10.8225368

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3740438, upper bound: 10.3740438
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3740438, upper bound: 10.3740438
time: 2.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1931313, upper bound: 10.1931313
time: 3.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1931313, upper bound: 10.1931313
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5261807, upper bound: 10.5261807
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5261807, upper bound: 10.5261807
time: 1.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
time: 1.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
time: 1.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319331, upper bound: 10.7319325
time: 3.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319325
time: 1.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
time: 2.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6960476, upper bound: 10.6960476
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6960476, upper bound: 10.6960476
time: 1.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7367881, upper bound: 10.7367881
time: 2.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7367881, upper bound: 10.7367881
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8449634, upper bound: 10.8449634
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8449634, upper bound: 10.8449634
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7171854, upper bound: 10.7171852
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7171854, upper bound: 10.7171852
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6941107, upper bound: 10.6941107
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6941107, upper bound: 10.6941107
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3106892, upper bound: 10.3106891
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3106892, upper bound: 10.3106891
time: 1.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454221, upper bound: 10.8454224
time: 2.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454221, upper bound: 10.8454224
time: 2.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6239114, upper bound: 10.6239121
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6239114, upper bound: 10.6239121
time: 3.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7805990, upper bound: 10.7805994
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7805990, upper bound: 10.7805994
time: 2.69 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.3740438, upper bound: 10.3740438
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.3740438, upper bound: 10.3740438
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.1931313, upper bound: 10.1931313
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.1931313, upper bound: 10.1931313
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.5261807, upper bound: 10.5261807
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.5261807, upper bound: 10.5261807
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7319331, upper bound: 10.7319325
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319325
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6960476, upper bound: 10.6960476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6960476, upper bound: 10.6960476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7367881, upper bound: 10.7367881
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7367881, upper bound: 10.7367881
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.8449634, upper bound: 10.8449634
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.8449634, upper bound: 10.8449634
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7171854, upper bound: 10.7171852
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7171854, upper bound: 10.7171852
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6941107, upper bound: 10.6941107
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6941107, upper bound: 10.6941107
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.3106892, upper bound: 10.3106891
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.3106892, upper bound: 10.3106891
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.8454221, upper bound: 10.8454224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.8454221, upper bound: 10.8454224
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6239114, upper bound: 10.6239121
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.6239114, upper bound: 10.6239121
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7805990, upper bound: 10.7805994
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 8, lower bound: -10.7805990, upper bound: 10.7805994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1832230, upper bound: 10.1832228
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1832228, upper bound: 10.1832230
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1832230, upper bound: 10.1832228
time: 1.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1832228, upper bound: 10.1832230
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
time: 2.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4457643, upper bound: 10.4457628
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4457643, upper bound: 10.4457628
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3963976, upper bound: 10.3963994
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3963976, upper bound: 10.3963994
time: 2.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
time: 1.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319331, upper bound: 10.7319324
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319331, upper bound: 10.7319325
time: 1.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319324
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319325
time: 9.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2042504, upper bound: 10.2042504
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2042504, upper bound: 10.2042504
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6820476, upper bound: 10.6820476
time: 2.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6820476, upper bound: 10.6820476
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4267526, upper bound: 10.4267526
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4267528, upper bound: 10.4267526
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6028211, upper bound: 10.6028213
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6028211, upper bound: 10.6028213
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7101077, upper bound: 10.7101073
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7101077, upper bound: 10.7101073
time: 9.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8398623, upper bound: 10.8398623
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8398623, upper bound: 10.8398623
time: 2.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7077931, upper bound: 10.7077928
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7077928, upper bound: 10.7077931
time: 2.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4414805, upper bound: 10.4414791
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4414799, upper bound: 10.4414791
time: 1.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5491094, upper bound: 10.5491094
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5491094, upper bound: 10.5491094
time: 3.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1933833, upper bound: 10.1933833
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1933833, upper bound: 10.1933833
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
time: 1.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5931669, upper bound: 10.5931664
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5931669, upper bound: 10.5931664
time: 2.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6743666, upper bound: 10.6743666
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.6743666, upper bound: 10.6743666
time: 2.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1027747, upper bound: 10.1027751
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1027747, upper bound: 10.1027751
time: 1.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1365181, upper bound: 10.1365179
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1365181, upper bound: 10.1365179
time: 3.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8279359, upper bound: 10.8279358
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8279359, upper bound: 10.8279358
time: 3.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454217, upper bound: 10.8454224
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.8454221, upper bound: 10.8454220
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3096391, upper bound: 10.3096398
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.3096391, upper bound: 10.3096398
time: 1.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3857970, upper bound: 10.3857974
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3857970, upper bound: 10.3857974
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5834205, upper bound: 10.5834210
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5834208, upper bound: 10.5834204
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5215616, upper bound: 10.5215621
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5215616, upper bound: 10.5215621
time: 1.78 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 6.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1832230, upper bound: 10.1832228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1832228, upper bound: 10.1832230
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1832230, upper bound: 10.1832228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1832228, upper bound: 10.1832230
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4457643, upper bound: 10.4457628
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4457643, upper bound: 10.4457628
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.3963976, upper bound: 10.3963994
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.3963976, upper bound: 10.3963994
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4645438, upper bound: 10.4645438
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7319331, upper bound: 10.7319324
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7319331, upper bound: 10.7319325
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319324
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319325
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.2042504, upper bound: 10.2042504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.2042504, upper bound: 10.2042504
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6820476, upper bound: 10.6820476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6820476, upper bound: 10.6820476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4267526, upper bound: 10.4267526
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4267528, upper bound: 10.4267526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6028211, upper bound: 10.6028213
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6028211, upper bound: 10.6028213
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7101077, upper bound: 10.7101073
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7101077, upper bound: 10.7101073
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.8398623, upper bound: 10.8398623
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.8398623, upper bound: 10.8398623
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7077931, upper bound: 10.7077928
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.7077928, upper bound: 10.7077931
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4414805, upper bound: 10.4414791
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.4414799, upper bound: 10.4414791
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5491094, upper bound: 10.5491094
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5491094, upper bound: 10.5491094
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1933833, upper bound: 10.1933833
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1933833, upper bound: 10.1933833
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5931669, upper bound: 10.5931664
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5931669, upper bound: 10.5931664
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6743666, upper bound: 10.6743666
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.6743666, upper bound: 10.6743666
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1027747, upper bound: 10.1027751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1027747, upper bound: 10.1027751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1365181, upper bound: 10.1365179
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.1365181, upper bound: 10.1365179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.8279359, upper bound: 10.8279358
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.8279359, upper bound: 10.8279358
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.8454217, upper bound: 10.8454224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.8454221, upper bound: 10.8454220
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.3096391, upper bound: 10.3096398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.3096391, upper bound: 10.3096398
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.3857970, upper bound: 10.3857974
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.3857970, upper bound: 10.3857974
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5834205, upper bound: 10.5834210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5834208, upper bound: 10.5834204
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5215616, upper bound: 10.5215621
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 6.93
Output dim: 8, lower bound: -10.5215616, upper bound: 10.5215621

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0799252, upper bound: 10.0799252
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.0799252, upper bound: 10.0799252
time: 1.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2386136, upper bound: 10.2386131
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2386137, upper bound: 10.2386131
time: 1.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2846504, upper bound: 10.2846504
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2846507, upper bound: 10.2846504
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
time: 1.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 214

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4287675, upper bound: 10.4287675
time: 1.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.4287675, upper bound: 10.4287675
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1454746, upper bound: 10.1454746
time: 1.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.1454746, upper bound: 10.1454746
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2876549, upper bound: 10.2876549
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -10.2876549, upper bound: 10.2876549
time: 1.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5290776, upper bound: 10.5290776
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5290776, upper bound: 10.5290776
time: 1.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 92

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319330, upper bound: 10.7319324
time: 1.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319324
time: 1.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5961585, upper bound: 10.5961585
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5961585, upper bound: 10.5961585
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -5.1619492, 4.2683043, -5.1619492, 4.2683043, -9.4302521, 9.4302540
1: -4.5150156, 4.0219212, -4.5150156, 4.0219212, -8.5369358, 8.5369358
2: -5.8974509, 4.1410961, -5.8974509, 4.1410961, -10.0385437, 10.0385447
3: -6.4188118, 3.7199407, -6.4188118, 3.7199407, -10.1387520, 10.1387520
4: -6.0448523, 4.4138484, -6.0448523, 4.4138484, -10.4587002, 10.4587002
5: -5.1020646, 4.2303286, -5.1020646, 4.2303286, -9.3323936, 9.3323936
6: -4.8802671, 4.7729130, -4.8802671, 4.7729130, -9.6531792, 9.6531792
7: -5.2257671, 5.0835156, -5.2257671, 5.0835156, -10.3092823, 10.3092823
8: -7.9439735, 4.0656700, -7.9439735, 4.0656700, -12.0096407, 12.0096416
9: -4.5706940, 4.8500509, -4.5706940, 4.8500509, -9.4207449, 9.4207449

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 126
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5949171, upper bound: 10.5949164
time: 8.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -10.5949171, upper bound: 10.5949164
time: 1.88 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 14.31 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.4710841, upper bound: 10.4710841
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.0799252, upper bound: 10.0799252
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.0799252, upper bound: 10.0799252
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.2386136, upper bound: 10.2386131
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.2386137, upper bound: 10.2386131
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.2846504, upper bound: 10.2846504
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.2846507, upper bound: 10.2846504
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.3842274, upper bound: 10.3842274
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.4287675, upper bound: 10.4287675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.4287675, upper bound: 10.4287675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.1454746, upper bound: 10.1454746
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.1454746, upper bound: 10.1454746
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.2876549, upper bound: 10.2876549
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.2876549, upper bound: 10.2876549
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.5290776, upper bound: 10.5290776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.5290776, upper bound: 10.5290776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.7319330, upper bound: 10.7319324
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.7319334, upper bound: 10.7319324
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.5961585, upper bound: 10.5961585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.5961585, upper bound: 10.5961585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.5949171, upper bound: 10.5949164
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 14.31
Output dim: 8, lower bound: -10.5949171, upper bound: 10.5949164
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.6616268, upper bound: 10.6616268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.6820476, upper bound: 10.6820476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.6820476, upper bound: 10.6820476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.4267526, upper bound: 10.4267526
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.4267528, upper bound: 10.4267526
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.6028211, upper bound: 10.6028213
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.6028211, upper bound: 10.6028213
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.7101077, upper bound: 10.7101073
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.7101077, upper bound: 10.7101073
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.8398623, upper bound: 10.8398623
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.8398623, upper bound: 10.8398623
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.7077931, upper bound: 10.7077928
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.7077928, upper bound: 10.7077931
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.4414805, upper bound: 10.4414791
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.4414799, upper bound: 10.4414791
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5491094, upper bound: 10.5491094
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5491094, upper bound: 10.5491094
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5011824, upper bound: 10.5011824
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5931669, upper bound: 10.5931664
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5931669, upper bound: 10.5931664
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.6743666, upper bound: 10.6743666
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.6743666, upper bound: 10.6743666
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.8279359, upper bound: 10.8279358
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.8279359, upper bound: 10.8279358
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.8454217, upper bound: 10.8454224
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.8454221, upper bound: 10.8454220
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.3857970, upper bound: 10.3857974
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.3857970, upper bound: 10.3857974
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5834205, upper bound: 10.5834210
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5834208, upper bound: 10.5834204
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5215616, upper bound: 10.5215621
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.31
Output dim: 8, lower bound: -10.5215616, upper bound: 10.5215621
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=12.009641647338867
rel_dist={8: [-10.853314892972527, 10.853314942404314]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1808.45 seconds
