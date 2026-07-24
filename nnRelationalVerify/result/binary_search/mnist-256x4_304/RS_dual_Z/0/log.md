## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 27.1733048946
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570)
1: (-16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204)
2: (-27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886)
3: (-24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909)
4: (-24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434)
5: (-18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721)
6: (-19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750)
7: (-22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555)
8: (-25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507)
9: (-17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555)

## BASE Result
execution time: IAR + LP analysis = 1.18 + 16.33 = 17.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2006483, upper bound: 27.2006483


# Binary Search by BASE starts (time budget: 1982.49 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=36.862388610839844
rel_dist={2: [-27.200554238179546, 27.20055423068736]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=36.862388610839844
rel_dist={2: [-27.20050538908776, 27.200505386462353]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=36.862388610839844
rel_dist={2: [-27.200440708568877, 27.200440707169562]}

## Binary Search Result
Binary search time: 28.61 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1953.88 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1988318, upper bound: 27.1988136
time: 19.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1988136, upper bound: 27.1988318
time: 5.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 25.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 25.35
Output dim: 2, lower bound: -27.1988318, upper bound: 27.1988136
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 25.35
Output dim: 2, lower bound: -27.1988136, upper bound: 27.1988318

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977583, upper bound: 27.1977023
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977061, upper bound: 27.1977347
time: 7.93 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977347, upper bound: 27.1977061
time: 2.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977023, upper bound: 27.1977583
time: 11.90 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.05
Output dim: 2, lower bound: -27.1977583, upper bound: 27.1977023
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.05
Output dim: 2, lower bound: -27.1977061, upper bound: 27.1977347
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.05
Output dim: 2, lower bound: -27.1977347, upper bound: 27.1977061
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.05
Output dim: 2, lower bound: -27.1977023, upper bound: 27.1977583

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977582, upper bound: 27.1977023
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977583, upper bound: 27.1977020
time: 11.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977061, upper bound: 27.1977347
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977061, upper bound: 27.1977345
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977345, upper bound: 27.1977061
time: 34.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977347, upper bound: 27.1977061
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977020, upper bound: 27.1977583
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1977023, upper bound: 27.1977582
time: 21.42 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.27 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977582, upper bound: 27.1977023
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977583, upper bound: 27.1977020
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977061, upper bound: 27.1977347
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977061, upper bound: 27.1977345
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977345, upper bound: 27.1977061
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977347, upper bound: 27.1977061
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977020, upper bound: 27.1977583
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.27
Output dim: 2, lower bound: -27.1977023, upper bound: 27.1977582

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918673
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918673
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918787, upper bound: 27.1918650
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918787, upper bound: 27.1918650
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918661, upper bound: 27.1918767
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918661, upper bound: 27.1918767
time: 3.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918681, upper bound: 27.1918741
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918681, upper bound: 27.1918741
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918741, upper bound: 27.1918681
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918741, upper bound: 27.1918681
time: 3.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918661
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918661
time: 6.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918650, upper bound: 27.1918787
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918650, upper bound: 27.1918787
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918673, upper bound: 27.1918766
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918673, upper bound: 27.1918766
time: 5.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 10.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918673
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918673
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918787, upper bound: 27.1918650
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918787, upper bound: 27.1918650
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918661, upper bound: 27.1918767
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918661, upper bound: 27.1918767
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918681, upper bound: 27.1918741
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918681, upper bound: 27.1918741
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918741, upper bound: 27.1918681
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918741, upper bound: 27.1918681
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918661
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918766, upper bound: 27.1918661
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918650, upper bound: 27.1918787
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918650, upper bound: 27.1918787
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918673, upper bound: 27.1918766
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.59
Output dim: 2, lower bound: -27.1918673, upper bound: 27.1918766

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
time: 3.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
time: 3.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755262
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
time: 3.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
time: 12.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
time: 8.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
time: 21.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
time: 21.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
time: 2.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
time: 2.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
time: 2.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
time: 2.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
time: 2.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755267
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755267
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755266
time: 3.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755266
time: 3.12 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755194
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755262
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755266
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.52
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755266

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
time: 7.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
time: 22.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
time: 21.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
time: 5.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
time: 3.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
time: 18.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755261
time: 3.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755261
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755262
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755262
time: 3.97 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755266, upper bound: 27.1755198
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755203
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755267, upper bound: 27.1755186
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755264, upper bound: 27.1755194
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755262
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755192, upper bound: 27.1755261
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.49
Output dim: 2, lower bound: -27.1755183, upper bound: 27.1755262
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755263
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755263, upper bound: 27.1755203
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755261, upper bound: 27.1755192
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755194, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755267
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755266
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.49
Output dim: 2, lower bound: -27.1755203, upper bound: 27.1755266
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=36.862388610839844
rel_dist={2: [-27.200554238179546, 27.20055423068736]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987630
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987630, upper bound: 27.1987716
time: 16.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 22.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 22.44
Output dim: 2, lower bound: -27.1987716, upper bound: 27.1987630
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 22.44
Output dim: 2, lower bound: -27.1987630, upper bound: 27.1987716

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976835
time: 12.16 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
time: 10.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968
time: 3.91 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976835
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.85
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976667
time: 4.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
time: 5.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
time: 7.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976667, upper bound: 27.1976968
time: 18.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968
time: 5.94 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976668
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976968, upper bound: 27.1976667
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976679, upper bound: 27.1976834
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976834, upper bound: 27.1976679
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976667, upper bound: 27.1976968
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.69
Output dim: 2, lower bound: -27.1976668, upper bound: 27.1976968

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
time: 12.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
time: 9.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
time: 15.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
time: 10.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
time: 6.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
time: 2.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
time: 5.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442
time: 6.42 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918442, upper bound: 27.1918391
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918450, upper bound: 27.1918380
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918388, upper bound: 27.1918436
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918398, upper bound: 27.1918427
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918427, upper bound: 27.1918398
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918436, upper bound: 27.1918388
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918380, upper bound: 27.1918450
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.76
Output dim: 2, lower bound: -27.1918391, upper bound: 27.1918442

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 2.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
time: 7.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 11.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 12.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 8.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 8.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 9.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
time: 6.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
time: 3.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 9.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 12.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
time: 15.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 13.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 15.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 3.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
time: 4.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 9.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754806
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 9.32
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 13.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 10.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
time: 5.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 3.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
time: 2.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 3.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 14.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754801
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
time: 6.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754853, upper bound: 27.1754807
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754851, upper bound: 27.1754811
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754856, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754801
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754806
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754805, upper bound: 27.1754854
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754851
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754810, upper bound: 27.1754852
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754852, upper bound: 27.1754810
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754854, upper bound: 27.1754805
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754806, upper bound: 27.1754856
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.20
Output dim: 2, lower bound: -27.1754811, upper bound: 27.1754853
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=36.862388610839844
rel_dist={2: [-27.20050538908776, 27.200505386462353]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987051, upper bound: 27.1987051
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1987051, upper bound: 27.1987067
time: 5.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.76
Output dim: 2, lower bound: -27.1987051, upper bound: 27.1987051
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.76
Output dim: 2, lower bound: -27.1987051, upper bound: 27.1987067

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976352, upper bound: 27.1976261
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976264, upper bound: 27.1976316
time: 5.09 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976316, upper bound: 27.1976264
time: 9.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976261, upper bound: 27.1976352
time: 12.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.01
Output dim: 2, lower bound: -27.1976352, upper bound: 27.1976261
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.01
Output dim: 2, lower bound: -27.1976264, upper bound: 27.1976316
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.01
Output dim: 2, lower bound: -27.1976316, upper bound: 27.1976264
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.01
Output dim: 2, lower bound: -27.1976261, upper bound: 27.1976352

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976312, upper bound: 27.1976261
time: 6.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976316, upper bound: 27.1976255
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976255, upper bound: 27.1976316
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976255, upper bound: 27.1976312
time: 5.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976312, upper bound: 27.1976264
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976316, upper bound: 27.1976260
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976255, upper bound: 27.1976352
time: 16.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1976261, upper bound: 27.1976352
time: 3.91 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976312, upper bound: 27.1976261
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976316, upper bound: 27.1976255
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976255, upper bound: 27.1976316
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976255, upper bound: 27.1976312
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976312, upper bound: 27.1976264
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976316, upper bound: 27.1976260
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976255, upper bound: 27.1976352
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.25
Output dim: 2, lower bound: -27.1976261, upper bound: 27.1976352

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918028
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918044, upper bound: 27.1918028
time: 24.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918046, upper bound: 27.1918022
time: 11.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918022
time: 5.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918040
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918040
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918036
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918030, upper bound: 27.1918036
time: 6.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918030
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918030
time: 3.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918040, upper bound: 27.1918025
time: 16.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918040, upper bound: 27.1918025
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918046
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918046
time: 6.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918028, upper bound: 27.1918044
time: 34.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1918028, upper bound: 27.1918044
time: 5.09 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 41.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918028
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918044, upper bound: 27.1918028
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918046, upper bound: 27.1918022
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918022
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918040
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918040
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918036
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918030, upper bound: 27.1918036
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918030
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918036, upper bound: 27.1918030
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918040, upper bound: 27.1918025
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918040, upper bound: 27.1918025
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918046
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918022, upper bound: 27.1918046
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918028, upper bound: 27.1918044
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 41.20
Output dim: 2, lower bound: -27.1918028, upper bound: 27.1918044

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
time: 7.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
time: 6.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
time: 6.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
time: 3.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 2.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 2.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 2.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 2.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
time: 14.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
time: 6.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
time: 14.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
time: 11.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
time: 9.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
time: 5.43 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.54 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754277, upper bound: 27.1754261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.54
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
time: 6.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 8.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754258
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
time: 6.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754258
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
time: 7.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754258
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -20.7441998, 15.8020573, -20.7441998, 15.8020573, -36.5462532, 36.5462570
1: -16.6654358, 14.0918903, -16.6654358, 14.0918903, -30.7573223, 30.7573204
2: -27.2343006, 9.6280870, -27.2343006, 9.6280870, -36.8623848, 36.8623886
3: -24.6729698, 11.2894220, -24.6729698, 11.2894220, -35.9623909, 35.9623909
4: -24.7506828, 15.1801605, -24.7506828, 15.1801605, -39.9308434, 39.9308434
5: -18.6637077, 15.9828682, -18.6637077, 15.9828682, -34.6465645, 34.6465721
6: -19.8873863, 16.9607925, -19.8873863, 16.9607925, -36.8481712, 36.8481750
7: -22.2695312, 16.4308281, -22.2695312, 16.4308281, -38.7003555, 38.7003555
8: -25.7181702, 14.7784910, -25.7181702, 14.7784910, -40.4966545, 40.4966507
9: -17.9596100, 20.8685474, -17.9596100, 20.8685474, -38.8281555, 38.8281555

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754256, upper bound: 27.1754258
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
time: 4.02 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 11.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754263
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754258
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754258
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754258
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754256, upper bound: 27.1754258
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.82
Output dim: 2, lower bound: -27.1754274, upper bound: 27.1754261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754276
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754275
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754275, upper bound: 27.1754263
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754276, upper bound: 27.1754261
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754261, upper bound: 27.1754277
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.82
Output dim: 2, lower bound: -27.1754263, upper bound: 27.1754276
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=36.862388610839844
rel_dist={2: [-27.200440708568877, 27.200440707169562]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1818.57 seconds
