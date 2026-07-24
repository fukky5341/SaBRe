## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.1888011651
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.7080693, 3.7080693)
1: (-11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.7309284, 3.7309284)
2: (-10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.9900298, 3.9900298)
3: (-5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009)
4: (-11.4109173, -8.3298731, -11.4109173, -8.3298731, -3.0810442, 3.0810442)
5: (6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.4367385, 2.4367385)
6: (-8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.5191064, 3.5191064)
7: (-17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.8375950, 3.8375950)
8: (-6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8985281, 2.8985281)
9: (-4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4910650, 2.4910650)

## BASE Result
execution time: IAR + LP analysis = 14.56 + 38.10 = 52.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8881189, upper bound: 1.8881190


# Binary Search by BASE starts (time budget: 3547.34 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2874808311462402
rel_dist={5: [-1.4581874533617398, 1.4581870675102389]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.1325697898864746
rel_dist={5: [-1.1923778842711146, 1.1923772606866638]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.0292954444885254
rel_dist={5: [-0.9450640713513714, 0.9450642034848773]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.0809326171875
rel_dist={5: [-1.077443187687913, 1.0774448872888032]}

## Binary Search Result
Binary search time: 240.46 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3306.88 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5367723, upper bound: 1.5356027
time: 7.68 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5356030, upper bound: 1.5367743
time: 14.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 22.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 22.57
Output dim: 5, lower bound: -1.5367723, upper bound: 1.5356027
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 22.57
Output dim: 5, lower bound: -1.5356030, upper bound: 1.5367743

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6669207, 3.6669207
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3730984, 3.3730989
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8770132, 3.8770099
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8944826, 2.8944874
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3391175, 2.3391199
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2394056, 3.2394052
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5518217, 3.5518260
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8918571, 2.8918538
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4765015, 2.4765024

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5365516, upper bound: 1.5355843
time: 11.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5367536, upper bound: 1.5353852
time: 8.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6669207, 3.6669202
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3730984, 3.3730989
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8770094, 3.8770127
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8944864, 2.8944826
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3391199, 2.3391175
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2394056, 3.2394061
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5518255, 3.5518212
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8918533, 2.8918576
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4765024, 2.4765019

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5350006, upper bound: 1.5306940
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5295231, upper bound: 1.5361401
time: 7.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 31.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.15
Output dim: 5, lower bound: -1.5365516, upper bound: 1.5355843
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.15
Output dim: 5, lower bound: -1.5367536, upper bound: 1.5353852
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 31.15
Output dim: 5, lower bound: -1.5350006, upper bound: 1.5306940
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 31.15
Output dim: 5, lower bound: -1.5295231, upper bound: 1.5361401

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6672258, 3.6673722
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3707314, 3.3714209
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8762445, 3.8760242
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8972483, 2.8981438
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3388088, 2.3386178
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2376633, 3.2396359
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5490437, 3.5498662
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8868208, 2.8881087
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4782305, 2.4776859

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5365515, upper bound: 1.5355356
time: 9.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5365062, upper bound: 1.5355836
time: 10.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6673708, 3.6672268
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3714209, 3.3707318
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8760281, 3.8762417
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8981390, 2.8972535
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3386161, 2.3388107
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2396364, 3.2376623
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5498610, 3.5490484
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8881130, 2.8868175
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4776850, 2.4782312

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5367497, upper bound: 1.5342049
time: 10.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5344081, upper bound: 1.5342116
time: 8.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6670475, 3.6669173
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3730488, 3.3730788
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8771143, 3.8770108
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8944845, 2.8945446
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3391175, 2.3391623
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2393265, 3.2393761
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5520267, 3.5518169
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8918514, 2.8919032
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4765015, 2.4765401

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5349897, upper bound: 1.5267234
time: 10.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5255489, upper bound: 1.5269901
time: 7.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6669178, 3.6670470
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3730793, 3.3730478
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8770075, 3.8771176
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8945494, 2.8944802
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3391647, 2.3391151
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2393751, 3.2393279
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5518208, 3.5520215
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8918991, 2.8918550
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4765415, 2.4765007

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5295029, upper bound: 1.5342305
time: 7.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5276087, upper bound: 1.5361210
time: 13.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 36.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5365515, upper bound: 1.5355356
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5365062, upper bound: 1.5355836
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5367497, upper bound: 1.5342049
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5344081, upper bound: 1.5342116
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5349897, upper bound: 1.5267234
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5255489, upper bound: 1.5269901
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5295029, upper bound: 1.5342305
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 36.00
Output dim: 5, lower bound: -1.5276087, upper bound: 1.5361210

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6667604, 3.6698132
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3619137, 3.3651733
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8682756, 3.8714914
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8987489, 2.8992662
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3538022, 2.3497272
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2154193, 3.2238665
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5404358, 3.5377383
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8931704, 2.8965919
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4821405, 2.4829128

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5365476, upper bound: 1.5343571
time: 14.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5342070, upper bound: 1.5343634
time: 7.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6696672, 3.6669064
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3644848, 3.3626032
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8717117, 3.8680549
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8983712, 2.8996449
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3499179, 2.3536115
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2218938, 3.2173915
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5369167, 3.5412583
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8953047, 2.8944585
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4834585, 2.4815955

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5364857, upper bound: 1.5336695
time: 22.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5345919, upper bound: 1.5355635
time: 11.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6627092, 3.6606464
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3442907, 3.3324418
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8715639, 3.8656802
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8779774, 2.8829684
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3337719, 2.3304965
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2385817, 3.2329941
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5532379, 3.5540404
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8845382, 2.8817739
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4554281, 2.4624634

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4999308, upper bound: 1.5341758
time: 9.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5367192, upper bound: 1.4973872
time: 42.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6607914, 3.6625633
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3331308, 3.3435979
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8654661, 3.8717775
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8838549, 2.8770919
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3303015, 2.3339620
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2349691, 3.2366076
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5548534, 3.5524249
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8830695, 2.8832436
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4619169, 2.4559743

Time for backsubstitution: 15.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 863

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5344080, upper bound: 1.5341634
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5343591, upper bound: 1.5342116
time: 9.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6687069, 3.6738396
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3775349, 3.3762965
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8623505, 3.8671966
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7103605, 2.7233725
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8970852, 2.8937030
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3159807, 2.3056002
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2562432, 3.2515049
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5207167, 3.5296278
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8902760, 2.8920851
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4789991, 2.4800241

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5349896, upper bound: 1.5266595
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5349514, upper bound: 1.5267237
time: 10.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6737099, 3.6685762
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3762665, 3.3775034
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8670826, 3.8622470
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7231026, 2.7100945
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8936434, 2.8970165
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3055556, 2.3159301
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2514558, 3.2561944
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5294266, 3.5205078
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8920326, 2.8903277
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4799852, 2.4790382

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5253285, upper bound: 1.5269706
time: 13.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5255295, upper bound: 1.5267695
time: 19.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6662283, 3.6685228
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3720779, 3.3751736
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8765526, 3.8780866
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8944025, 2.8947940
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3440714, 2.3368099
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2398930, 3.2390838
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5508175, 3.5541492
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8913059, 2.8931251
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4759693, 2.4777188

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4926827, upper bound: 1.5342012
time: 23.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5294725, upper bound: 1.4974307
time: 7.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6669178, 3.6663570
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3730793, 3.3720465
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8770075, 3.8766623
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8945494, 2.8943329
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3368597, 2.3391151
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2391310, 3.2393279
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5518208, 3.5510173
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8918991, 2.8912623
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4765415, 2.4759290

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5264368, upper bound: 1.5338347
time: 11.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5264303, upper bound: 1.5361172
time: 8.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 34.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5365476, upper bound: 1.5343571
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5342070, upper bound: 1.5343634
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5364857, upper bound: 1.5336695
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5345919, upper bound: 1.5355635
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.4999308, upper bound: 1.5341758
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5367192, upper bound: 1.4973872
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5344080, upper bound: 1.5341634
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5343591, upper bound: 1.5342116
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5349896, upper bound: 1.5266595
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5349514, upper bound: 1.5267237
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5253285, upper bound: 1.5269706
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5255295, upper bound: 1.5267695
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.4926827, upper bound: 1.5342012
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5294725, upper bound: 1.4974307
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5264368, upper bound: 1.5338347
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.15
Output dim: 5, lower bound: -1.5264303, upper bound: 1.5361172

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6620989, 3.6632328
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3347836, 3.3268838
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8638134, 3.8609324
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8785868, 2.8849814
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3489580, 2.3414128
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2143645, 3.2191982
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5438099, 3.5427279
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8895969, 2.8915484
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4598832, 2.4671440

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5364699, upper bound: 1.5244920
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5266580, upper bound: 1.5343408
time: 7.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6601801, 3.6651502
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3236237, 3.3380404
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8577156, 3.8670292
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8844643, 2.8791044
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3454881, 2.3448782
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2107501, 3.2228117
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5454254, 3.5411124
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8881264, 2.8930182
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4663720, 2.4606550

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4973869, upper bound: 1.5343328
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5341765, upper bound: 1.4975449
time: 12.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6689787, 3.6683836
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3634815, 3.3647285
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8712587, 3.8690252
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8982234, 2.8999581
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3548231, 2.3513052
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2224126, 3.2171476
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5359097, 3.5433846
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8947115, 2.8957283
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4828854, 2.4828131

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5364162, upper bound: 1.5238167
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5266046, upper bound: 1.5336289
time: 14.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6696672, 3.6662178
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3644848, 3.3616014
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8717117, 3.8676009
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8983712, 2.8994970
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3476119, 2.3536115
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2216506, 3.2173915
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5369167, 3.5402527
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8953047, 2.8938653
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4834585, 2.4810233

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 863

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5062395, upper bound: 1.5355536
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5345818, upper bound: 1.5072223
time: 8.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6618452, 3.6594253
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3305941, 3.3101878
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8852129, 3.8754721
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8847561, 2.8924026
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.2615643, 2.2793331
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2029123, 3.2077079
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5525818, 3.5531135
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8608179, 2.8482804
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4413137, 2.4425354

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4998756, upper bound: 1.5243310
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4900606, upper bound: 1.5341549
time: 8.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6614885, 3.6597810
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3220358, 3.3187461
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8813562, 3.8793287
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8874111, 2.8897481
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.2826080, 2.2582889
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2132959, 3.1973243
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5523109, 3.5533834
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8510447, 2.8580534
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4355011, 2.4483488

Time for backsubstitution: 14.89 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.339118003845215
rel_dist={5: [-1.5369944972060878, 1.5369941521841248]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 5777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805990, upper bound: 1.2867787
time: 53.53 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867765, upper bound: 1.2805992
time: 7.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 61.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 61.21
Output dim: 5, lower bound: -1.2805990, upper bound: 1.2867787
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 61.21
Output dim: 5, lower bound: -1.2867765, upper bound: 1.2805992

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4340591, 3.4360967
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1026649, 3.1040092
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6681719, 3.6720600
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.5222640, 2.5162244
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6564932, 2.6590352
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1754184, 2.1814685
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9520245, 2.9559865
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2445898, 3.2453318
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7117720, 2.7064095
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3694458, 2.3704767

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805377, upper bound: 1.2809485
time: 12.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747702, upper bound: 1.2867151
time: 9.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4360962, 3.4340587
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1040096, 3.1026645
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6720600, 3.6681719
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.5162244, 2.5222633
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6590357, 2.6564932
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1814685, 2.1754186
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9559860, 2.9520254
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2453318, 3.2445893
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7064095, 2.7117720
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3704777, 2.3694468

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867646, upper bound: 1.2786537
time: 19.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2848317, upper bound: 1.2805872
time: 10.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 44.18 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 44.18
Output dim: 5, lower bound: -1.2805377, upper bound: 1.2809485
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 44.18
Output dim: 5, lower bound: -1.2747702, upper bound: 1.2867151
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 44.18
Output dim: 5, lower bound: -1.2867646, upper bound: 1.2786537
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 44.18
Output dim: 5, lower bound: -1.2848317, upper bound: 1.2805872

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4359245, 3.4409709
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1066055, 3.1072254
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6534081, 3.6601243
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4766941, 2.4782424
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6577215, 2.6582961
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1478126, 2.1479056
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9668880, 2.9681139
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2132826, 3.2192364
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7101974, 2.7058392
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3719435, 2.3735380

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805373, upper bound: 1.2804361
time: 11.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2800238, upper bound: 1.2809478
time: 23.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4389324, 3.4379630
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1058807, 3.1079502
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6562366, 3.6572957
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4842815, 2.4706550
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6557541, 2.6602635
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1418555, 2.1538630
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9641528, 2.9708495
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2184944, 3.2140245
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7112017, 2.7048354
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3725071, 2.3729744

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2537666, upper bound: 1.2866969
time: 13.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747540, upper bound: 1.2657136
time: 8.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4354095, 3.4346089
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1030064, 3.1034493
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6716061, 3.6685314
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.5163774, 2.5220668
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6588869, 2.6566086
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1832848, 2.1731133
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9561777, 2.9517813
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2443256, 3.2453737
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7058163, 2.7122433
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3699055, 2.3698974

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867035, upper bound: 1.2728270
time: 9.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2809363, upper bound: 1.2785932
time: 9.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4360962, 3.4333711
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1040096, 3.1016622
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6720600, 3.6677175
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.5160284, 2.5222633
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6590357, 2.6563449
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1791630, 2.1754186
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9557428, 2.9520254
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2453318, 3.2435837
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7064095, 2.7111788
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3704777, 2.3688745

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2847713, upper bound: 1.2747583
time: 9.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2790051, upper bound: 1.2805262
time: 17.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 42.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2805373, upper bound: 1.2804361
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2800238, upper bound: 1.2809478
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2537666, upper bound: 1.2866969
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2747540, upper bound: 1.2657136
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2867035, upper bound: 1.2728270
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2809363, upper bound: 1.2785932
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2847713, upper bound: 1.2747583
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 42.18
Output dim: 5, lower bound: -1.2790051, upper bound: 1.2805262

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4355068, 3.4406228
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0979166, 3.0999823
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6521254, 3.6599650
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4772735, 2.4787226
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6601162, 2.6611848
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1476521, 2.1477714
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9622602, 2.9625597
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2064037, 3.2135043
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7099361, 2.7055407
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3735809, 2.3755126

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2801897, upper bound: 1.2803928
time: 16.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2748734, upper bound: 1.2803872
time: 9.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4355764, 3.4405527
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0993633, 3.0985360
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6532488, 3.6588416
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4771743, 2.4788215
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6606102, 2.6606917
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1476789, 2.1477447
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9613342, 2.9634857
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2075500, 3.2123580
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7098989, 2.7055774
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3739195, 2.3751750

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2800119, upper bound: 1.2790044
time: 9.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2780797, upper bound: 1.2809359
time: 17.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4379139, 3.4367418
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0885181, 3.0856967
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6682377, 3.6670933
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4842825, 2.4707317
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6625328, 2.6685586
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0696468, 2.0936735
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9284763, 2.9411063
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2177229, 3.2130985
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6832886, 2.6713428
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3559017, 2.3530474

Time for backsubstitution: 15.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 863

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2534525, upper bound: 1.2866906
time: 15.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2534628, upper bound: 1.2654295
time: 11.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4377098, 3.4369454
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0836277, 3.0905900
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6660337, 3.6692982
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4843588, 2.4706559
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6640491, 2.6670423
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0816722, 2.0816543
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9344101, 2.9351726
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2175684, 3.2132530
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6777086, 2.6769273
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3525801, 2.3563709

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747539, upper bound: 1.2656752
time: 23.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747130, upper bound: 1.2657134
time: 15.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4372740, 3.4394803
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1069479, 3.1066647
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6568413, 3.6565957
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4708080, 2.4840853
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6601162, 2.6558704
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1556773, 2.1395493
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9710412, 2.9639096
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2130194, 3.2192788
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7042418, 2.7116733
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3724031, 2.3729591

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867032, upper bound: 1.2723155
time: 18.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2861897, upper bound: 1.2728266
time: 26.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4402809, 3.4364729
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1062231, 3.1073895
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6596699, 3.6537676
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4783955, 2.4764979
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6581488, 2.6578379
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1497207, 2.1455066
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9683061, 2.9666452
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2182322, 3.2140675
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7052460, 2.7106693
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3729658, 2.3723955

Time for backsubstitution: 15.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2806103, upper bound: 1.2777380
time: 26.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2800756, upper bound: 1.2782740
time: 9.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4379635, 3.4382429
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1079502, 3.1048775
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6572952, 3.6557822
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4704590, 2.4842813
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6602631, 2.6556072
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1515565, 2.1418555
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9706054, 2.9641528
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2140255, 3.2174892
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7048349, 2.7106087
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3729744, 2.3719358

Time for backsubstitution: 14.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2847694, upper bound: 1.2735782
time: 11.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2835901, upper bound: 1.2747565
time: 10.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4409695, 3.4352350
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1072254, 3.1056027
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6601238, 3.6529536
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4780464, 2.4766939
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6582966, 2.6575742
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1455989, 2.1478128
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9678702, 2.9668884
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2192364, 3.2122779
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7058392, 2.7096047
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3735380, 2.3713727

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2790036, upper bound: 1.2805235
time: 10.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2790031, upper bound: 1.2805266
time: 13.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 38.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2801897, upper bound: 1.2803928
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2748734, upper bound: 1.2803872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2800119, upper bound: 1.2790044
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2780797, upper bound: 1.2809359
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2534525, upper bound: 1.2866906
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2534628, upper bound: 1.2654295
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2747539, upper bound: 1.2656752
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2747130, upper bound: 1.2657134
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2867032, upper bound: 1.2723155
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2861897, upper bound: 1.2728266
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2806103, upper bound: 1.2777380
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2800756, upper bound: 1.2782740
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2847694, upper bound: 1.2735782
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2835901, upper bound: 1.2747565
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2790036, upper bound: 1.2805235
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 38.73
Output dim: 5, lower bound: -1.2790031, upper bound: 1.2805266

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4353714, 3.4404135
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0978680, 3.0999508
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6521835, 3.6599631
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4774151, 2.4787123
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6600113, 2.6611166
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1476512, 2.1477969
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9621820, 2.9625077
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2065134, 3.2134967
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7099342, 2.7055666
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3735800, 2.3755341

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 5777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2801896, upper bound: 1.2803494
time: 10.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2801510, upper bound: 1.2803904
time: 14.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4352970, 3.4403391
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0978498, 3.0999336
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6521225, 3.6598997
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4772635, 2.4785583
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6599751, 2.6610799
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1476235, 2.1477699
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9621534, 2.9624801
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2063961, 3.2133789
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7099609, 2.7055387
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3736029, 2.3755112

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2748615, upper bound: 1.2784437
time: 10.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2729303, upper bound: 1.2803751
time: 8.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4348869, 3.4411016
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0983610, 3.0993214
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6527929, 3.6591997
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4773269, 2.4786255
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6604633, 2.6608076
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1494942, 2.1454394
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9615254, 2.9632416
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2065458, 3.2131433
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7093058, 2.7060487
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3733473, 2.3756261

Time for backsubstitution: 14.67 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.18420672416687
rel_dist={5: [-1.286786395541573, 1.2867863462497207]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1873650, upper bound: 1.1923705
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923705, upper bound: 1.1873645
time: 11.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.57
Output dim: 5, lower bound: -1.1873650, upper bound: 1.1923705
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.57
Output dim: 5, lower bound: -1.1923705, upper bound: 1.1873645

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3574276, 3.3589568
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0131741, 3.0141816
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6004419, 3.6033573
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4438605, 2.4393315
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5783920, 2.5802984
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1237817, 2.1283190
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8581371, 2.8611088
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1425400, 3.1430960
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6512127, 2.6471915
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3342609, 2.3350334

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 863

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1758206, upper bound: 1.1923660
time: 12.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1873606, upper bound: 1.1808180
time: 13.82 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3589573, 3.3574281
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0141811, 3.0131731
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6033573, 3.6004410
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4393315, 2.4438608
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5802984, 2.5783920
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1283193, 2.1237817
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8611088, 2.8581381
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1430960, 3.1425390
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6471920, 2.6512134
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3350334, 2.3342605

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923688, upper bound: 1.1873630
time: 14.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923684, upper bound: 1.1873632
time: 10.02 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 38.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 38.86
Output dim: 5, lower bound: -1.1758206, upper bound: 1.1923660
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 38.86
Output dim: 5, lower bound: -1.1873606, upper bound: 1.1808180
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 38.86
Output dim: 5, lower bound: -1.1923688, upper bound: 1.1873630
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 38.86
Output dim: 5, lower bound: -1.1923684, upper bound: 1.1873632

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3554878, 3.3567395
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0096512, 3.0101552
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6017342, 3.6043949
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4438581, 2.4395108
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5808659, 2.5833693
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1074839, 2.1140583
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8508778, 2.8547564
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1429672, 3.1436205
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6447091, 2.6397562
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3306208, 2.3308740

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1712569, upper bound: 1.1923511
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1712854, upper bound: 1.1762675
time: 32.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3593483, 3.3578820
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0118151, 3.0111012
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6025887, 3.5995798
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4396982, 2.4441831
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5830650, 2.5815401
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1280105, 2.1233897
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8604932, 2.8583689
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1403198, 3.1401124
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6428938, 2.6474695
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3367624, 2.3357553

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1920349, upper bound: 1.1849377
time: 22.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1899419, upper bound: 1.1870212
time: 11.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3594112, 3.3578196
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0121107, 3.0108061
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6024961, 3.5996728
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4396534, 2.4442277
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5834475, 2.5811586
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1279275, 2.1234722
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8613400, 2.8575230
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1406689, 3.1397624
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6434469, 2.6469162
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3365278, 2.3359890

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1919581, upper bound: 1.1866619
time: 11.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1916744, upper bound: 1.1869460
time: 6.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 32.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 5, lower bound: -1.1712569, upper bound: 1.1923511
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 32.58
Output dim: 5, lower bound: -1.1712854, upper bound: 1.1762675
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 5, lower bound: -1.1920349, upper bound: 1.1849377
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 5, lower bound: -1.1899419, upper bound: 1.1870212
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 5, lower bound: -1.1919581, upper bound: 1.1866619
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 5, lower bound: -1.1916744, upper bound: 1.1869460

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3544197, 3.3555188
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9950867, 2.9919233
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6131830, 3.6141915
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4438567, 2.4395714
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5876436, 2.5912890
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0352769, 2.0508673
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8151994, 2.8235288
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1421547, 3.1426926
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6154037, 2.6062660
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3131847, 2.3109460

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 4636

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1712033, upper bound: 1.1920069
time: 20.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1709124, upper bound: 1.1922961
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3594017, 3.3578801
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0117655, 3.0110645
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6026306, 3.5995770
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4398012, 2.4441724
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5830622, 2.5815654
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1280086, 2.1234086
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8604150, 2.8583107
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1403999, 3.1401067
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6428919, 2.6474881
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3367596, 2.3357706

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 5777

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1920269, upper bound: 1.1829780
time: 18.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1900770, upper bound: 1.1849292
time: 12.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3593464, 3.3579359
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0117769, 3.0110517
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6025848, 3.5996232
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4396877, 2.4442866
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5830898, 2.5815377
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1280286, 2.1233883
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8604360, 2.8582897
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1403122, 3.1401944
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6429129, 2.6474674
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3367767, 2.3357534

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 863

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1899411, upper bound: 1.1870088
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1899419, upper bound: 1.1870215
time: 10.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3594103, 3.3578191
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0121088, 3.0108051
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6024914, 3.5996675
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4396520, 2.4442253
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5834398, 2.5811539
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1279275, 2.1234729
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8613377, 2.8575211
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1406689, 3.1397634
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6434426, 2.6469107
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3365278, 2.3359880

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1919092, upper bound: 1.1822390
time: 9.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1875388, upper bound: 1.1866154
time: 10.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3594103, 3.3578186
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0121088, 3.0108051
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6024904, 3.5996690
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4396520, 2.4442253
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5834417, 2.5811520
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1279285, 2.1234722
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8613377, 2.8575220
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1406708, 3.1397614
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6434426, 2.6469121
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3365278, 2.3359876

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 444

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1913370, upper bound: 1.1845157
time: 17.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1892500, upper bound: 1.1866121
time: 18.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 50.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1712033, upper bound: 1.1920069
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1709124, upper bound: 1.1922961
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1920269, upper bound: 1.1829780
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1900770, upper bound: 1.1849292
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1899411, upper bound: 1.1870088
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1899419, upper bound: 1.1870215
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1919092, upper bound: 1.1822390
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1875388, upper bound: 1.1866154
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1913370, upper bound: 1.1845157
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 50.69
Output dim: 5, lower bound: -1.1892500, upper bound: 1.1866121

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3540001, 3.3551521
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9863968, 2.9843183
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6119003, 3.6137509
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4444118, 2.4400518
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5900402, 2.5940552
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0351171, 2.0507276
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8103399, 2.8179736
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1352758, 3.1366735
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6151323, 2.6059675
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3148217, 2.3128357

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1712017, upper bound: 1.1920049
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1712014, upper bound: 1.1920052
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3540525, 3.3550997
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9874811, 2.9832335
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6127424, 3.6129084
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4443374, 2.4401262
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5904102, 2.5936852
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0351372, 2.0507076
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8096447, 2.8186688
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1361361, 3.1358142
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6151056, 2.6059949
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3150735, 2.3125825

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1709042, upper bound: 1.1903377
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1689533, upper bound: 1.1922877
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3587132, 3.3581185
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0107622, 3.0114031
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6021767, 3.5997324
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4398670, 2.4439762
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5829153, 2.5816159
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1287928, 2.1211026
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8604984, 2.8580675
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1393967, 3.1404443
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6422987, 2.6476936
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3361883, 2.3359661

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1920216, upper bound: 1.1808857
time: 13.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1878433, upper bound: 1.1810000
time: 15.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3594017, 3.3571901
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0117655, 3.0100627
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6026306, 3.5991220
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4396057, 2.4441724
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5830622, 2.5814180
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1257019, 2.1234086
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8601723, 2.8583107
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1403999, 3.1391020
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6428919, 2.6468954
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3367596, 2.3351994

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 863

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1785384, upper bound: 1.1849249
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1900725, upper bound: 1.1733834
time: 11.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3588810, 3.3587155
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0029602, 3.0033355
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5946169, 3.5931268
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4472685, 2.4509146
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5843749, 2.5826602
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1408029, 2.1344976
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8381901, 2.8388200
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1296930, 3.1280661
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6492629, 2.6547318
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3406858, 2.3402267

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1899329, upper bound: 1.1850507
time: 28.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1879816, upper bound: 1.1870014
time: 10.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3601265, 3.3574696
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0040607, 3.0022340
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5960903, 3.5916538
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4463158, 2.4518676
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5842128, 2.5828223
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1391377, 2.1361623
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8409653, 2.8360453
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1281843, 3.1295748
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6501765, 2.6538174
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3412504, 2.3396626

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1899337, upper bound: 1.1850635
time: 11.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1879824, upper bound: 1.1870155
time: 32.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3612766, 3.3619413
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0158701, 3.0140228
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5877285, 3.5870256
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3940811, 2.4043455
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5841761, 2.5804148
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0988312, 2.0899091
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8755178, 2.8696494
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1093607, 3.1123643
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6418695, 2.6460891
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3390265, 2.3389101

Time for backsubstitution: 15.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 6136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1919013, upper bound: 1.1802788
time: 11.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1899517, upper bound: 1.1822295
time: 11.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3594637, 3.3578167
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0120611, 3.0107694
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6025333, 3.5996671
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4397550, 2.4442146
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5834398, 2.5811777
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1279266, 2.1234910
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8612585, 2.8574634
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1407528, 3.1397557
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6434398, 2.6469305
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3365259, 2.3360033

Time for backsubstitution: 14.95 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.1325697898864746
rel_dist={5: [-1.1923778842711146, 1.1923772606866638]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2429.49 seconds
