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
execution time: IAR + LP analysis = 1.18 + 16.16 = 17.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -27.2006483, upper bound: 27.2006483


# Binary Search by BASE starts (time budget: 1982.66 seconds, max iter: 100)

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
Binary search time: 28.48 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1954.18 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1925573, upper bound: 27.1925573
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1925573, upper bound: 27.1925573
time: 4.23 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.29 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 2, lower bound: -27.1925573, upper bound: 27.1925573
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.29
Output dim: 2, lower bound: -27.1925573, upper bound: 27.1925573

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1690687, upper bound: 27.1690687
time: 18.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1690687, upper bound: 27.1690687
time: 17.00 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802336, upper bound: 27.1802336
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802336, upper bound: 27.1802336
time: 8.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.02 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.02
Output dim: 2, lower bound: -27.1690687, upper bound: 27.1690687
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.02
Output dim: 2, lower bound: -27.1690687, upper bound: 27.1690687
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.02
Output dim: 2, lower bound: -27.1802336, upper bound: 27.1802336
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.02
Output dim: 2, lower bound: -27.1802336, upper bound: 27.1802336

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724066, upper bound: 27.1724066
time: 21.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724066, upper bound: 27.1724066
time: 18.52 seconds

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1801961, upper bound: 27.1801954
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1801961, upper bound: 27.1801954
time: 5.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.59 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 14.59
Output dim: 2, lower bound: -27.1724066, upper bound: 27.1724066
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 14.59
Output dim: 2, lower bound: -27.1724066, upper bound: 27.1724066
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.59
Output dim: 2, lower bound: -27.1801961, upper bound: 27.1801954
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.59
Output dim: 2, lower bound: -27.1801961, upper bound: 27.1801954

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742257
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742257
time: 4.93 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1654486, upper bound: 27.1654409
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1654486, upper bound: 27.1654409
time: 3.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 10.96 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 10.96
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742257
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 10.96
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742257
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 10.96
Output dim: 2, lower bound: -27.1654486, upper bound: 27.1654409
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 10.96
Output dim: 2, lower bound: -27.1654486, upper bound: 27.1654409

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1742071, upper bound: 27.1742257
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742118
time: 3.96 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742254
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742257
time: 3.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.02 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.02
Output dim: 2, lower bound: -27.1742071, upper bound: 27.1742257
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.02
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742118
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.02
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742254
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.02
Output dim: 2, lower bound: -27.1742099, upper bound: 27.1742257

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742079
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742079
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1619340, upper bound: 27.1619367
time: 9.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1619340, upper bound: 27.1619367
time: 6.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741993, upper bound: 27.1742089
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741894, upper bound: 27.1742170
time: 2.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1558403, upper bound: 27.1558449
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1558403, upper bound: 27.1558449
time: 8.18 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 17.10 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742079
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742079
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1619340, upper bound: 27.1619367
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1619340, upper bound: 27.1619367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1741993, upper bound: 27.1742089
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1741894, upper bound: 27.1742170
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1558403, upper bound: 27.1558449
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 17.10
Output dim: 2, lower bound: -27.1558403, upper bound: 27.1558449

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741796, upper bound: 27.1742000
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741796, upper bound: 27.1742000
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742070
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741841, upper bound: 27.1742079
time: 12.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1613894, upper bound: 27.1613653
time: 45.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1613894, upper bound: 27.1613655
time: 24.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741894, upper bound: 27.1742008
time: 6.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741794, upper bound: 27.1742171
time: 9.20 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 16.63 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1741796, upper bound: 27.1742000
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1741796, upper bound: 27.1742000
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742070
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1741841, upper bound: 27.1742079
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1613894, upper bound: 27.1613653
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1613894, upper bound: 27.1613655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1741894, upper bound: 27.1742008
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 16.63
Output dim: 2, lower bound: -27.1741794, upper bound: 27.1742171

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1619344, upper bound: 27.1619199
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1619344, upper bound: 27.1619199
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741788, upper bound: 27.1742000
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741796, upper bound: 27.1741988
time: 19.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741848, upper bound: 27.1742070
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742030
time: 11.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723419, upper bound: 27.1723627
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723419, upper bound: 27.1723627
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1613537, upper bound: 27.1613615
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1613537, upper bound: 27.1613612
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716152, upper bound: 27.1716486
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1716149, upper bound: 27.1716484
time: 3.89 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 9.22 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1619344, upper bound: 27.1619199
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1619344, upper bound: 27.1619199
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1741788, upper bound: 27.1742000
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1741796, upper bound: 27.1741988
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1741848, upper bound: 27.1742070
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742030
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1723419, upper bound: 27.1723627
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1723419, upper bound: 27.1723627
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1613537, upper bound: 27.1613615
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1613537, upper bound: 27.1613612
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1716152, upper bound: 27.1716486
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 9.22
Output dim: 2, lower bound: -27.1716149, upper bound: 27.1716484

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1614960, upper bound: 27.1615160
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1614960, upper bound: 27.1615158
time: 5.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1537173, upper bound: 27.1537077
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1537173, upper bound: 27.1537077
time: 3.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1725115, upper bound: 27.1725303
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1724990, upper bound: 27.1725346
time: 6.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741854, upper bound: 27.1742031
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742012
time: 5.64 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 10.86 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1614960, upper bound: 27.1615160
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1614960, upper bound: 27.1615158
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1537173, upper bound: 27.1537077
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1537173, upper bound: 27.1537077
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1725115, upper bound: 27.1725303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1724990, upper bound: 27.1725346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1741854, upper bound: 27.1742031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 10.86
Output dim: 2, lower bound: -27.1741863, upper bound: 27.1742012

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741724, upper bound: 27.1741902
time: 9.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1741668, upper bound: 27.1741939
time: 7.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1632404, upper bound: 27.1632347
time: 6.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1632404, upper bound: 27.1632347
time: 7.15 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 15.27 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 15.27
Output dim: 2, lower bound: -27.1741724, upper bound: 27.1741902
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 15.27
Output dim: 2, lower bound: -27.1741668, upper bound: 27.1741939
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 15.27
Output dim: 2, lower bound: -27.1632404, upper bound: 27.1632347
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 15.27
Output dim: 2, lower bound: -27.1632404, upper bound: 27.1632347

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1556944, upper bound: 27.1556859
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1556944, upper bound: 27.1556859
time: 9.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1617284, upper bound: 27.1617536
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1617284, upper bound: 27.1617536
time: 4.75 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 10.51 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 10.51
Output dim: 2, lower bound: -27.1556944, upper bound: 27.1556859
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 10.51
Output dim: 2, lower bound: -27.1556944, upper bound: 27.1556859
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 10.51
Output dim: 2, lower bound: -27.1617284, upper bound: 27.1617536
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 10.51
Output dim: 2, lower bound: -27.1617284, upper bound: 27.1617536
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=36.862388610839844
rel_dist={2: [-27.200554238179546, 27.20055423068736]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1950174, upper bound: 27.1950174
time: 3.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1950174, upper bound: 27.1950174
time: 3.24 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.40
Output dim: 2, lower bound: -27.1950174, upper bound: 27.1950174
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.40
Output dim: 2, lower bound: -27.1950174, upper bound: 27.1950174

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164
time: 4.08 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164
time: 2.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164
time: 6.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 10.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.29
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.29
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 10.29
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 10.29
Output dim: 2, lower bound: -27.1944164, upper bound: 27.1944164

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1865060, upper bound: 27.1865060
time: 3.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1865060, upper bound: 27.1865060
time: 3.66 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1899069, upper bound: 27.1899069
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1899069, upper bound: 27.1899069
time: 4.02 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1861326, upper bound: 27.1861326
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1861326, upper bound: 27.1861326
time: 11.63 seconds

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
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1849505, upper bound: 27.1849505
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1849505, upper bound: 27.1849505
time: 4.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 10.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1865060, upper bound: 27.1865060
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1865060, upper bound: 27.1865060
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1899069, upper bound: 27.1899069
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1899069, upper bound: 27.1899069
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1861326, upper bound: 27.1861326
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1861326, upper bound: 27.1861326
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1849505, upper bound: 27.1849505
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 10.63
Output dim: 2, lower bound: -27.1849505, upper bound: 27.1849505

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796654
time: 3.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796654
time: 6.55 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1794608, upper bound: 27.1794600
time: 8.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1794608, upper bound: 27.1794600
time: 4.63 seconds

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

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753977
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753976
time: 4.28 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753977
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753976
time: 4.20 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846250, upper bound: 27.1845550
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1845550, upper bound: 27.1846250
time: 3.21 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755680
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755680
time: 15.32 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1701734, upper bound: 27.1701734
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1701734, upper bound: 27.1701734
time: 2.81 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1739826, upper bound: 27.1739826
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1739826, upper bound: 27.1739826
time: 3.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796654
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796654
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1794608, upper bound: 27.1794600
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1794608, upper bound: 27.1794600
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753977
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753976
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753977
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1753977, upper bound: 27.1753976
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1846250, upper bound: 27.1845550
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1845550, upper bound: 27.1846250
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755680
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755680
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1701734, upper bound: 27.1701734
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1701734, upper bound: 27.1701734
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1739826, upper bound: 27.1739826
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.45
Output dim: 2, lower bound: -27.1739826, upper bound: 27.1739826

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796629, upper bound: 27.1796646
time: 8.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796629
time: 4.94 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796547
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796556, upper bound: 27.1796654
time: 5.48 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1605186, upper bound: 27.1605190
time: 30.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1605186, upper bound: 27.1605190
time: 30.89 seconds

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1794529, upper bound: 27.1794600
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1794608, upper bound: 27.1794521
time: 6.41 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723286, upper bound: 27.1723286
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1723286, upper bound: 27.1723286
time: 3.80 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753667, upper bound: 27.1753785
time: 9.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753784, upper bound: 27.1753668
time: 3.93 seconds

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
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753976, upper bound: 27.1753578
time: 8.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753580, upper bound: 27.1753977
time: 3.55 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1556992, upper bound: 27.1556992
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1556992, upper bound: 27.1556992
time: 5.14 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846206, upper bound: 27.1845550
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846250, upper bound: 27.1845473
time: 8.15 seconds

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
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802810, upper bound: 27.1803076
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802810, upper bound: 27.1803076
time: 3.44 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755670
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755670, upper bound: 27.1755680
time: 5.33 seconds

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755679
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1755679, upper bound: 27.1755680
time: 4.81 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1706555, upper bound: 27.1706575
time: 3.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1706575, upper bound: 27.1706555
time: 3.26 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1739826, upper bound: 27.1739820
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1739820, upper bound: 27.1739826
time: 10.05 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 14.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1796629, upper bound: 27.1796646
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796629
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796547
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1796556, upper bound: 27.1796654
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1605186, upper bound: 27.1605190
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1605186, upper bound: 27.1605190
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1794529, upper bound: 27.1794600
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1794608, upper bound: 27.1794521
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1723286, upper bound: 27.1723286
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1723286, upper bound: 27.1723286
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1753667, upper bound: 27.1753785
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1753784, upper bound: 27.1753668
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1753976, upper bound: 27.1753578
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1753580, upper bound: 27.1753977
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1556992, upper bound: 27.1556992
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1556992, upper bound: 27.1556992
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1846206, upper bound: 27.1845550
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1846250, upper bound: 27.1845473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1802810, upper bound: 27.1803076
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1802810, upper bound: 27.1803076
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755670
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1755670, upper bound: 27.1755680
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1755680, upper bound: 27.1755679
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1755679, upper bound: 27.1755680
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1706555, upper bound: 27.1706575
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1706575, upper bound: 27.1706555
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1739826, upper bound: 27.1739820
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 2, lower bound: -27.1739820, upper bound: 27.1739826

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1631972, upper bound: 27.1632035
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1631972, upper bound: 27.1632035
time: 3.91 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796020
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796092, upper bound: 27.1796629
time: 3.95 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796556
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796654, upper bound: 27.1796547
time: 4.33 seconds

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796556, upper bound: 27.1796654
time: 18.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1796556, upper bound: 27.1796654
time: 5.29 seconds

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
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1774999, upper bound: 27.1775227
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1774999, upper bound: 27.1775227
time: 4.42 seconds

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

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1634794, upper bound: 27.1634648
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1634794, upper bound: 27.1634648
time: 3.85 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1580985, upper bound: 27.1580984
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1580985, upper bound: 27.1580984
time: 4.49 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1555863, upper bound: 27.1555665
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1555863, upper bound: 27.1555665
time: 3.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746882, upper bound: 27.1745996
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746882, upper bound: 27.1745996
time: 17.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753378, upper bound: 27.1753976
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1753580, upper bound: 27.1753813
time: 9.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1607362, upper bound: 27.1607406
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1607362, upper bound: 27.1607406
time: 7.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 217

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1831853, upper bound: 27.1831322
time: 10.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1831772, upper bound: 27.1831439
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802786, upper bound: 27.1803076
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1802810, upper bound: 27.1803076
time: 8.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1785351, upper bound: 27.1785602
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1785351, upper bound: 27.1785602
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1695350, upper bound: 27.1695352
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1695350, upper bound: 27.1695352
time: 5.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754510, upper bound: 27.1754594
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1754594, upper bound: 27.1754494
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=36.862388610839844
rel_dist={2: [-27.200601580158846, 27.20060156497138]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1925723, upper bound: 27.1925723
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1925723, upper bound: 27.1925723
time: 12.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 18.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 18.54
Output dim: 2, lower bound: -27.1925723, upper bound: 27.1925723
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 18.54
Output dim: 2, lower bound: -27.1925723, upper bound: 27.1925723

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1916793, upper bound: 27.1916793
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1916793, upper bound: 27.1916793
time: 3.20 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1821003, upper bound: 27.1821003
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1821003, upper bound: 27.1821003
time: 6.00 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.73
Output dim: 2, lower bound: -27.1916793, upper bound: 27.1916793
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.73
Output dim: 2, lower bound: -27.1916793, upper bound: 27.1916793
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.73
Output dim: 2, lower bound: -27.1821003, upper bound: 27.1821003
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.73
Output dim: 2, lower bound: -27.1821003, upper bound: 27.1821003

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
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 194

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846284, upper bound: 27.1846285
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846284, upper bound: 27.1846285
time: 2.83 seconds

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
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1875419, upper bound: 27.1875419
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1875419, upper bound: 27.1875419
time: 10.60 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807331, upper bound: 27.1806848
time: 10.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1806848, upper bound: 27.1807323
time: 6.00 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1821003, upper bound: 27.1820775
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1820783, upper bound: 27.1821003
time: 7.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1846284, upper bound: 27.1846285
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1846284, upper bound: 27.1846285
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1875419, upper bound: 27.1875419
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1875419, upper bound: 27.1875419
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1807331, upper bound: 27.1806848
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1806848, upper bound: 27.1807323
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1821003, upper bound: 27.1820775
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.59
Output dim: 2, lower bound: -27.1820783, upper bound: 27.1821003

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846285, upper bound: 27.1846267
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846268, upper bound: 27.1846285
time: 4.51 seconds

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
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1474490, upper bound: 27.1474490
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1474490, upper bound: 27.1474490
time: 3.95 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1875419, upper bound: 27.1875324
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1875324, upper bound: 27.1875419
time: 5.37 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1848200, upper bound: 27.1848200
time: 3.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1848200, upper bound: 27.1848200
time: 4.21 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807248, upper bound: 27.1806840
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807331, upper bound: 27.1806726
time: 4.14 seconds

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1696477, upper bound: 27.1696662
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1696477, upper bound: 27.1696662
time: 5.25 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1752085
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1752085
time: 6.87 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1801769, upper bound: 27.1801970
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1801689, upper bound: 27.1802040
time: 3.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1846285, upper bound: 27.1846267
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1846268, upper bound: 27.1846285
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1474490, upper bound: 27.1474490
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1474490, upper bound: 27.1474490
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1875419, upper bound: 27.1875324
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1875324, upper bound: 27.1875419
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1848200, upper bound: 27.1848200
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1848200, upper bound: 27.1848200
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1807248, upper bound: 27.1806840
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1807331, upper bound: 27.1806726
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1696477, upper bound: 27.1696662
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1696477, upper bound: 27.1696662
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1752085
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1752085
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1801769, upper bound: 27.1801970
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.59
Output dim: 2, lower bound: -27.1801689, upper bound: 27.1802040

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1717078, upper bound: 27.1716788
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1717078, upper bound: 27.1716788
time: 3.41 seconds

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846123, upper bound: 27.1846284
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1846267, upper bound: 27.1846145
time: 4.71 seconds

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
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 160

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
time: 3.42 seconds

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1749627, upper bound: 27.1749630
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1749627, upper bound: 27.1749630
time: 5.40 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1776304, upper bound: 27.1776344
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1776304, upper bound: 27.1776344
time: 3.04 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1783042, upper bound: 27.1783042
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1783042, upper bound: 27.1783042
time: 9.68 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807248, upper bound: 27.1806848
time: 29.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807240, upper bound: 27.1806838
time: 6.30 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1728534, upper bound: 27.1728299
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1728534, upper bound: 27.1728300
time: 4.18 seconds

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752075, upper bound: 27.1752085
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1752072
time: 3.70 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1739531, upper bound: 27.1739527
time: 3.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1739531, upper bound: 27.1739527
time: 3.08 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1580343, upper bound: 27.1580357
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1580343, upper bound: 27.1580357
time: 9.89 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1773647, upper bound: 27.1773808
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1773647, upper bound: 27.1773808
time: 5.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1717078, upper bound: 27.1716788
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1717078, upper bound: 27.1716788
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1846123, upper bound: 27.1846284
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1846267, upper bound: 27.1846145
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1749627, upper bound: 27.1749630
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1749627, upper bound: 27.1749630
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1776304, upper bound: 27.1776344
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1776304, upper bound: 27.1776344
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1783042, upper bound: 27.1783042
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1783042, upper bound: 27.1783042
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1807248, upper bound: 27.1806848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1807240, upper bound: 27.1806838
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1728534, upper bound: 27.1728299
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1728534, upper bound: 27.1728300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1752075, upper bound: 27.1752085
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1752072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1739531, upper bound: 27.1739527
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1739531, upper bound: 27.1739527
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1580343, upper bound: 27.1580357
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1580343, upper bound: 27.1580357
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1773647, upper bound: 27.1773808
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.71
Output dim: 2, lower bound: -27.1773647, upper bound: 27.1773808

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1786040, upper bound: 27.1786130
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1786040, upper bound: 27.1786139
time: 5.41 seconds

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

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1700372, upper bound: 27.1700050
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1700372, upper bound: 27.1700050
time: 2.94 seconds

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777521
time: 11.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1777575, upper bound: 27.1777607
time: 3.83 seconds

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

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
time: 4.28 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746162, upper bound: 27.1746162
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1746162, upper bound: 27.1746162
time: 4.68 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 103

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1749627, upper bound: 27.1749384
time: 21.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1749384, upper bound: 27.1749630
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 160

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1776304, upper bound: 27.1776338
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1776292, upper bound: 27.1776344
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1763232, upper bound: 27.1763053
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1763042, upper bound: 27.1763248
time: 8.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1663972, upper bound: 27.1663972
time: 13.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1663972, upper bound: 27.1663972
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1782942, upper bound: 27.1783042
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1783042, upper bound: 27.1782942
time: 3.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1792846, upper bound: 27.1792607
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1792846, upper bound: 27.1792607
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807240, upper bound: 27.1806838
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1807222, upper bound: 27.1806847
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1728498, upper bound: 27.1728356
time: 7.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -27.1728498, upper bound: 27.1728356
time: 14.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 103
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 160
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1751997
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -27.1752004, upper bound: 27.1752072
time: 42.86 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 50.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1786040, upper bound: 27.1786130
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1786040, upper bound: 27.1786139
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1700372, upper bound: 27.1700050
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1700372, upper bound: 27.1700050
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777521
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1777575, upper bound: 27.1777607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1777666, upper bound: 27.1777607
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1746162, upper bound: 27.1746162
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1746162, upper bound: 27.1746162
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1749627, upper bound: 27.1749384
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1749384, upper bound: 27.1749630
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1776304, upper bound: 27.1776338
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1776292, upper bound: 27.1776344
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1763232, upper bound: 27.1763053
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1763042, upper bound: 27.1763248
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1663972, upper bound: 27.1663972
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1663972, upper bound: 27.1663972
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1782942, upper bound: 27.1783042
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1783042, upper bound: 27.1782942
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1792846, upper bound: 27.1792607
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1792846, upper bound: 27.1792607
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1807240, upper bound: 27.1806838
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1807222, upper bound: 27.1806847
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1728498, upper bound: 27.1728356
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1728498, upper bound: 27.1728356
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1752086, upper bound: 27.1751997
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 50.97
Output dim: 2, lower bound: -27.1752004, upper bound: 27.1752072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.97
Output dim: 2, lower bound: -27.1739531, upper bound: 27.1739527
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.97
Output dim: 2, lower bound: -27.1739531, upper bound: 27.1739527
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 50.97
Output dim: 2, lower bound: -27.1773647, upper bound: 27.1773808
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 50.97
Output dim: 2, lower bound: -27.1773647, upper bound: 27.1773808
Binary search (step 2): status=Status.UNKNOWN, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=36.862388610839844
rel_dist={2: [-27.200570093667537, 27.2005701317934]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1745.63 seconds
