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
execution time: IAR + LP analysis = 14.33 + 37.20 = 51.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.8881189, upper bound: 1.8881190


# Binary Search by BASE starts (time budget: 3548.47 seconds, max iter: 100)

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
Binary search time: 241.05 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3307.42 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5280181, upper bound: 1.5369655
time: 12.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5369658, upper bound: 1.5280176
time: 12.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 25.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 25.37
Output dim: 5, lower bound: -1.5280181, upper bound: 1.5369655
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 25.37
Output dim: 5, lower bound: -1.5369658, upper bound: 1.5280176

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6639500, 3.6675167
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3711395, 3.3734918
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8713655, 3.8781686
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8907957, 2.8952446
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3303299, 2.3409173
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2336884, 3.2406192
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5507393, 3.5520382
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8934479, 2.8840630
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4750061, 2.4768081

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5279607, upper bound: 1.5271172
time: 8.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5181461, upper bound: 1.5369282
time: 11.48 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6669197, 3.6639490
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3730984, 3.3711386
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8770151, 3.8713651
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7289009
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8944883, 2.8907962
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3391180, 2.3303299
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2394066, 3.2336876
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5518208, 3.5507393
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8840637, 2.8918593
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4765034, 2.4750051

Time for backsubstitution: 14.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5369288, upper bound: 1.5181467
time: 10.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5271179, upper bound: 1.5279601
time: 9.71 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 35.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 35.34
Output dim: 5, lower bound: -1.5279607, upper bound: 1.5271172
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 35.34
Output dim: 5, lower bound: -1.5181461, upper bound: 1.5369282
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 35.34
Output dim: 5, lower bound: -1.5369288, upper bound: 1.5181467
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 35.34
Output dim: 5, lower bound: -1.5271179, upper bound: 1.5279601

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6658154, 3.6746459
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3756237, 3.3767080
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8566017, 3.8683538
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7119017, 2.7146113
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8934994, 2.8945060
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3071921, 2.3073542
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2506022, 3.2527466
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5194330, 3.5298519
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8918734, 2.8842459
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4775028, 2.4802916

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5279569, upper bound: 1.5247731
time: 12.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5256563, upper bound: 1.5271135
time: 22.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6710777, 3.6693826
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3743553, 3.3779769
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8615494, 3.8634048
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7251797, 2.7013330
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8900566, 2.8979487
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.2967665, 2.3177798
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2458148, 3.2575340
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5285530, 3.5207319
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8936300, 2.8824890
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4784880, 2.4793057

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5181423, upper bound: 1.5346021
time: 9.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5158191, upper bound: 1.5369247
time: 10.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6687851, 3.6710787
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3775845, 3.3743548
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8622513, 3.8615503
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7013330, 2.7233846
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8971920, 2.8900571
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3159792, 2.2967665
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2563224, 3.2458150
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5205145, 3.5285530
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8824892, 2.8920407
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4790001, 2.4784884

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5369250, upper bound: 1.5158189
time: 10.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5346029, upper bound: 1.5181426
time: 13.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6740494, 3.6658154
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3763142, 3.3756237
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8672009, 3.8566012
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7146111, 2.7101064
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8937492, 2.8934999
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3055537, 2.3071921
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2515349, 3.2506022
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5296354, 3.5194325
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8842459, 2.8902838
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4799862, 2.4775026

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5271141, upper bound: 1.5256557
time: 15.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5247734, upper bound: 1.5279563
time: 10.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 40.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5279569, upper bound: 1.5247731
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5256563, upper bound: 1.5271135
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5181423, upper bound: 1.5346021
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5158191, upper bound: 1.5369247
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5369250, upper bound: 1.5158189
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5346029, upper bound: 1.5181426
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5271141, upper bound: 1.5256557
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 40.57
Output dim: 5, lower bound: -1.5247734, upper bound: 1.5279563

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6611538, 3.6680660
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3484936, 3.3384185
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8521318, 3.8577876
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7097797, 2.7116191
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8733368, 2.8802209
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3023510, 2.2990434
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2495465, 3.2480779
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5228081, 3.5348425
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8882999, 2.8792028
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4552460, 2.4645233

Time for backsubstitution: 14.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5279568, upper bound: 1.5247055
time: 19.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5279189, upper bound: 1.5247752
time: 15.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6592350, 3.6699843
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3373337, 3.3495774
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8460340, 3.8638849
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7089090, 2.7124901
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8792152, 2.8743439
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.2988815, 2.3025134
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2459340, 3.2516913
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5244236, 3.5332274
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8868303, 2.8806727
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4617348, 2.4580345

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5256562, upper bound: 1.5270676
time: 10.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5256072, upper bound: 1.5271128
time: 14.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6664171, 3.6628027
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3472233, 3.3396873
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8570814, 3.8528380
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7230587, 2.6983409
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8698940, 2.8836632
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.2919254, 2.3094690
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2447591, 3.2528651
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5319281, 3.5257225
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8900576, 2.8774459
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4562311, 2.4635375

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5181422, upper bound: 1.5345534
time: 10.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5181057, upper bound: 1.5346027
time: 9.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6644993, 3.6647210
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3360653, 3.3508468
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8509836, 3.8589358
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7221870, 2.6992121
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8757725, 2.8777866
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.2884560, 2.3129389
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2411466, 3.2564785
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5335436, 3.5241070
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8885880, 2.8789158
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4627209, 2.4570484

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5158190, upper bound: 1.5368874
time: 13.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4790045, upper bound: 1.5368940
time: 8.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5157922, upper bound: 1.5001141
time: 14.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6641245, 3.6644988
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3504524, 3.3360653
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8577814, 3.8509841
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.6992121, 2.7203920
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8770294, 2.8757725
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3111391, 2.2884557
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2552657, 3.2411461
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5238905, 3.5335436
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8789158, 2.8869977
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4567432, 2.4627204

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5369249, upper bound: 1.5157406
time: 9.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5368867, upper bound: 1.5158186
time: 8.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6622057, 3.6664166
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3392935, 3.3472247
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8516855, 3.8570814
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.6983414, 2.7212629
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8829069, 2.8698950
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3076687, 2.2919259
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2516522, 3.2447596
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5255060, 3.5319285
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8774462, 2.8884676
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4632320, 2.4562314

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5346028, upper bound: 1.5181048
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5345539, upper bound: 1.5181413
time: 9.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6693869, 3.6592355
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3491831, 3.3373342
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8627310, 3.8460345
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7124901, 2.7071137
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8735876, 2.8792152
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3007135, 2.2988813
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2504783, 3.2459335
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5330095, 3.5244236
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8806725, 2.8852408
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4577293, 2.4617345

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5271140, upper bound: 1.5256069
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5270683, upper bound: 1.5256556
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6674690, 3.6611533
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3380241, 3.3484936
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8566332, 3.8521323
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7116194, 2.7079849
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8794641, 2.8733377
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.2972431, 2.3023515
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2468648, 3.2495470
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5346251, 3.5228081
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8792028, 2.8867106
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4642181, 2.4552455

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5247734, upper bound: 1.5279185
time: 9.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5247040, upper bound: 1.5279572
time: 12.39 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 36.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5279568, upper bound: 1.5247055
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5279189, upper bound: 1.5247752
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5256562, upper bound: 1.5270676
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5256072, upper bound: 1.5271128
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5181422, upper bound: 1.5345534
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5181057, upper bound: 1.5346027
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.4790045, upper bound: 1.5368940
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5157922, upper bound: 1.5001141
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5369249, upper bound: 1.5157406
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5368867, upper bound: 1.5158186
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5346028, upper bound: 1.5181048
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5345539, upper bound: 1.5181413
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5271140, upper bound: 1.5256069
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5270683, upper bound: 1.5256556
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5247734, upper bound: 1.5279185
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 5, lower bound: -1.5247040, upper bound: 1.5279572

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6606846, 3.6705041
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3396759, 3.3321705
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8441639, 3.8532567
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7186322, 2.7182484
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8748374, 2.8813426
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3173442, 2.3101523
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2273030, 3.2323077
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5142012, 3.5227141
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8946486, 2.8876843
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4591541, 2.4697499

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4911463, upper bound: 1.5246735
time: 10.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5279308, upper bound: 1.4879069
time: 16.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6635914, 3.6675973
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3422461, 3.3296003
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8476009, 3.8498201
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7164092, 2.7204719
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8744597, 2.8817208
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3134604, 2.3140364
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2337766, 3.2258329
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5106802, 3.5262346
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8967819, 2.8855510
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4604721, 2.4684327

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4911076, upper bound: 1.5247426
time: 11.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5278929, upper bound: 1.4879567
time: 12.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6587658, 3.6724224
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3285160, 3.3433294
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8380680, 3.8593540
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7177615, 2.7191193
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8807158, 2.8754656
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3138747, 2.3136222
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2236886, 3.2359214
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5158148, 3.5210991
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8931780, 2.8891540
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4656429, 2.4632611

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4888315, upper bound: 1.5270372
time: 14.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5256293, upper bound: 1.4902539
time: 9.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6616745, 3.6695151
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3310871, 3.3407593
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8415031, 3.8559175
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7155375, 2.7213428
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8803363, 2.8758438
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3099904, 2.3175063
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2301641, 3.2294464
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5122957, 3.5246196
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8953123, 2.8870208
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4669609, 2.4619439

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.4887837, upper bound: 1.5270834
time: 11.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.5255803, upper bound: 1.4902995
time: 8.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.6659489, 3.6652408
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.3384066, 3.3334394
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.8491135, 3.8483071
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.7289009, 2.7049701
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.8713946, 2.8847849
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.3069186, 2.3205776
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -3.2225156, 3.2370951
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.5233202, 3.5135942
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.8964052, 2.8859274
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.4601402, 2.4687641

Time for backsubstitution: 14.65 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.339118003845215
rel_dist={5: [-1.5369944972060878, 1.5369941521841248]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6182
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805990, upper bound: 1.2867787
time: 53.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867765, upper bound: 1.2805992
time: 6.88 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 61.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 61.07
Output dim: 5, lower bound: -1.2805990, upper bound: 1.2867787
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 61.07
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

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805377, upper bound: 1.2809485
time: 12.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747702, upper bound: 1.2867151
time: 9.73 seconds

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

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867154, upper bound: 1.2747700
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2809482, upper bound: 1.2805374
time: 12.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 36.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 36.31
Output dim: 5, lower bound: -1.2805377, upper bound: 1.2809485
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 36.31
Output dim: 5, lower bound: -1.2747702, upper bound: 1.2867151
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 36.31
Output dim: 5, lower bound: -1.2867154, upper bound: 1.2747700
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 36.31
Output dim: 5, lower bound: -1.2809482, upper bound: 1.2805374

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

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805358, upper bound: 1.2797732
time: 15.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2793562, upper bound: 1.2809461
time: 10.69 seconds

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

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747683, upper bound: 1.2855366
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2735904, upper bound: 1.2867135
time: 14.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4379635, 3.4389319
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1079502, 3.1058807
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6572952, 3.6562362
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4706545, 2.4842813
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6602631, 2.6557546
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1538627, 2.1418555
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9708495, 2.9641528
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2140255, 3.2184944
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7048349, 2.7112019
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3729744, 2.3725071

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867135, upper bound: 1.2735900
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2855343, upper bound: 1.2747682
time: 9.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4409695, 3.4359245
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.1072254, 3.1066060
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6601238, 3.6534081
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4782419, 2.4766939
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6582966, 2.6577215
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1479056, 2.1478128
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9681144, 2.9668884
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2192364, 3.2132826
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7058392, 2.7101979
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3735380, 2.3719440

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2809463, upper bound: 1.2793558
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2797717, upper bound: 1.2805356
time: 11.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 37.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2805358, upper bound: 1.2797732
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2793562, upper bound: 1.2809461
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2747683, upper bound: 1.2855366
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2735904, upper bound: 1.2867135
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2867135, upper bound: 1.2735900
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2855343, upper bound: 1.2747682
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2809463, upper bound: 1.2793558
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 37.36
Output dim: 5, lower bound: -1.2797717, upper bound: 1.2805356

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4304409, 3.4343910
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0746937, 3.0689359
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6463251, 3.6495576
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4741993, 2.4752502
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6375589, 2.6414928
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1414852, 2.1395948
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9642844, 2.9634447
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2166586, 3.2235351
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7059946, 2.7007961
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3496866, 2.3549886

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805357, upper bound: 1.2797291
time: 9.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2804971, upper bound: 1.2797717
time: 14.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4293451, 3.4354868
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0683155, 3.0753121
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6428404, 3.6530423
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4737015, 2.4757481
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6409168, 2.6381340
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1395016, 2.1415775
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9622197, 2.9655099
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2175808, 3.2226119
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7051544, 2.7016361
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3533955, 2.3512807

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2793560, upper bound: 1.2809055
time: 13.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2793161, upper bound: 1.2809461
time: 11.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4334488, 3.4313831
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0739670, 3.0696607
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6491537, 3.6467299
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4817867, 2.4676628
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6355915, 2.6434598
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1355276, 2.1455522
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9615483, 2.9661803
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2218695, 3.2183237
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7069988, 2.6997924
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3502502, 2.3544252

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747682, upper bound: 1.2854957
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2747274, upper bound: 1.2855347
time: 8.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4323521, 3.4324789
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0675907, 3.0760379
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6456690, 3.6502137
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4812889, 2.4681604
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6389503, 2.6401014
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1335449, 2.1475351
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9594836, 2.9682450
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2227926, 3.2174006
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7061586, 2.7006321
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3539581, 2.3507173

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2735903, upper bound: 1.2866749
time: 26.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2735487, upper bound: 1.2867132
time: 10.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4324799, 3.4323521
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0760365, 3.0675912
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6502123, 3.6456704
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4681606, 2.4812891
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6401005, 2.6389503
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1475353, 2.1335447
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9682450, 2.9594836
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2174006, 3.2227926
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7006321, 2.7061589
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3507175, 2.3539584

Time for backsubstitution: 15.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2867134, upper bound: 1.2735488
time: 6.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2866749, upper bound: 1.2735903
time: 10.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4313831, 3.4334483
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0696602, 3.0739679
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6467295, 3.6491542
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4676628, 2.4817870
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6434593, 2.6355925
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1455517, 2.1355276
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9661803, 2.9615488
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2183237, 3.2218695
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6997929, 2.7069988
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3544254, 2.3502502

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2855342, upper bound: 1.2747268
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2854955, upper bound: 1.2747684
time: 18.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4354868, 3.4293447
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0753117, 3.0683165
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6530409, 3.6428418
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4757481, 2.4737017
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6381340, 2.6409178
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1415777, 2.1395020
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9655099, 2.9622197
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2226114, 3.2175808
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7016363, 2.7051549
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3512802, 2.3533950

Time for backsubstitution: 14.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2809462, upper bound: 1.2793163
time: 12.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2809054, upper bound: 1.2793560
time: 12.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4343910, 3.4304409
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0689354, 3.0746932
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6495581, 3.6463261
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4752502, 2.4741993
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6414928, 2.6375594
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1395950, 2.1414850
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9634442, 2.9642844
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2235346, 3.2166581
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7007961, 2.7059948
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3549881, 2.3496869

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2797719, upper bound: 1.2804973
time: 8.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2797291, upper bound: 1.2805358
time: 12.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 36.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2805357, upper bound: 1.2797291
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2804971, upper bound: 1.2797717
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2793560, upper bound: 1.2809055
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2793161, upper bound: 1.2809461
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2747682, upper bound: 1.2854957
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2747274, upper bound: 1.2855347
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2735903, upper bound: 1.2866749
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2735487, upper bound: 1.2867132
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2867134, upper bound: 1.2735488
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2866749, upper bound: 1.2735903
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2855342, upper bound: 1.2747268
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2854955, upper bound: 1.2747684
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2809462, upper bound: 1.2793163
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2809054, upper bound: 1.2793560
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2797719, upper bound: 1.2804973
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 36.44
Output dim: 5, lower bound: -1.2797291, upper bound: 1.2805358

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4299717, 3.4355831
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0658751, 3.0615864
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6383591, 3.6435542
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4820991, 2.4818795
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6388984, 2.6426146
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1548133, 2.1507034
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9420400, 2.9449000
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2065420, 3.2114067
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7123423, 2.7083633
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3535957, 2.3596506

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2595286, upper bound: 1.2797100
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2805194, upper bound: 1.2587320
time: 11.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4316330, 3.4339218
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0673437, 3.0601177
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6403217, 3.6415906
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4808288, 2.4831500
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6386809, 2.6428301
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1525941, 2.1529231
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9457402, 2.9411998
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2045298, 3.2134185
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7135620, 2.7071443
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3543491, 2.3588979

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2594900, upper bound: 1.2797530
time: 11.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2804809, upper bound: 1.2587706
time: 44.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4288750, 3.4366789
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0594978, 3.0679631
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6348743, 3.6470385
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4816012, 2.4823773
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6422553, 2.6392558
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1528306, 2.1526864
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9399743, 2.9469647
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2074652, 3.2104836
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7115030, 2.7092032
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3573036, 2.3559427

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2583486, upper bound: 1.2808870
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.2793397, upper bound: 1.2599068
time: 10.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.4305363, 3.4350181
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0609665, 3.0664940
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.6368389, 3.6450744
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.4803309, 2.4836478
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.6420398, 2.6394722
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1506109, 2.1549060
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.9436746, 2.9432645
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.2054529, 3.2124953
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.7127218, 2.7079842
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3580570, 2.3551900

Time for backsubstitution: 14.57 seconds
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
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6182

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1873650, upper bound: 1.1923705
time: 10.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923705, upper bound: 1.1873645
time: 11.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 22.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 22.13
Output dim: 5, lower bound: -1.1873650, upper bound: 1.1923705
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 22.13
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

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1873159, upper bound: 1.1879482
time: 12.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1829354, upper bound: 1.1873160
time: 34.11 seconds

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

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 444
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 444

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923224, upper bound: 1.1829350
time: 9.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1879484, upper bound: 1.1873159
time: 10.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 34.81 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 34.81
Output dim: 5, lower bound: -1.1873159, upper bound: 1.1879482
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 34.81
Output dim: 5, lower bound: -1.1829354, upper bound: 1.1873160
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 34.81
Output dim: 5, lower bound: -1.1923224, upper bound: 1.1829350
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 34.81
Output dim: 5, lower bound: -1.1879484, upper bound: 1.1873159

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3608227, 3.3615499
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0179424, 3.0163894
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5885925, 3.5877986
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3937616, 2.4039819
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5810356, 2.5776534
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0992239, 2.0902183
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8752875, 2.8702655
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1117897, 3.1151409
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6456175, 2.6503923
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3375311, 2.3371801

Time for backsubstitution: 15.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923211, upper bound: 1.1821503
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915433, upper bound: 1.1829339
time: 14.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 38.84 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.84
Output dim: 5, lower bound: -1.1923211, upper bound: 1.1821503
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.84
Output dim: 5, lower bound: -1.1915433, upper bound: 1.1829339

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3550653, 3.3549700
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9844341, 2.9780998
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5806398, 3.5772319
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3911438, 2.4009898
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5608730, 2.5600100
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0924006, 2.0819075
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8721681, 2.8655963
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1151648, 3.1192093
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6412048, 2.6453493
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3152733, 2.3177042

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923210, upper bound: 1.1821379
time: 20.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923181, upper bound: 1.1821527
time: 19.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3542433, 3.3557920
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9796524, 2.9828820
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5780268, 3.5798454
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3907700, 2.4013631
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5633907, 2.5574913
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0909128, 2.0833948
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8706193, 2.8671451
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1158571, 3.1185169
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6405745, 2.6459792
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3180542, 2.3149233

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6136
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6136

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915432, upper bound: 1.1829214
time: 12.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915396, upper bound: 1.1829340
time: 12.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 40.37 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 40.37
Output dim: 5, lower bound: -1.1923210, upper bound: 1.1821379
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 40.37
Output dim: 5, lower bound: -1.1923181, upper bound: 1.1821527
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 40.37
Output dim: 5, lower bound: -1.1915432, upper bound: 1.1829214
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 40.37
Output dim: 5, lower bound: -1.1915396, upper bound: 1.1829340

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3545971, 3.3557467
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9756165, 2.9703832
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5726719, 3.5707374
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3987250, 2.4076190
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5621572, 2.5611317
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1051736, 2.0930166
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8499227, 2.8461266
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1045446, 3.1070809
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6475525, 2.6526117
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3191833, 2.3221781

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1765386, upper bound: 1.1821252
time: 12.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923066, upper bound: 1.1663732
time: 8.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3558426, 3.3545012
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9767179, 2.9692817
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5741463, 3.5692649
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3977723, 2.4085720
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5619960, 2.5612938
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1035094, 2.0946813
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8526978, 2.8433518
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1030359, 3.1085892
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6484671, 2.6516974
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3197479, 2.3216136

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1765357, upper bound: 1.1821379
time: 8.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923037, upper bound: 1.1663861
time: 19.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3537750, 3.3565688
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9708338, 2.9751658
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5700588, 3.5733504
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3983521, 2.4079924
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5646758, 2.5586131
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1036868, 2.0945036
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8483739, 2.8476753
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1052370, 3.1063886
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6469221, 2.6532416
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3219643, 2.3193972

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1757608, upper bound: 1.1829107
time: 17.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915288, upper bound: 1.1671591
time: 8.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3550205, 3.3553233
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9719353, 2.9740639
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5715332, 3.5718780
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3973994, 2.4089453
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5645137, 2.5587752
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.1020222, 2.0961683
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8511491, 2.8449001
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1037283, 3.1078973
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6478367, 2.6523273
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3225288, 2.3188326

Time for backsubstitution: 14.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5777
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5777

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1757579, upper bound: 1.1829210
time: 13.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915252, upper bound: 1.1671692
time: 9.97 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 38.62 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1765386, upper bound: 1.1821252
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1923066, upper bound: 1.1663732
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1765357, upper bound: 1.1821379
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1923037, upper bound: 1.1663861
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1757608, upper bound: 1.1829107
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1915288, upper bound: 1.1671591
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1757579, upper bound: 1.1829210
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 38.62
Output dim: 5, lower bound: -1.1915252, upper bound: 1.1671692

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3533783, 3.3546810
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9533644, 2.9517989
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5824680, 3.5821853
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3987827, 2.4076190
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5700717, 2.5679097
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0419803, 2.0208087
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8186941, 2.8104477
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1036196, 3.1062708
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6140604, 2.6233044
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.2992554, 2.3047419

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1922560, upper bound: 1.1660227
time: 19.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1919605, upper bound: 1.1663072
time: 14.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3546238, 3.3534350
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9544668, 2.9506974
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5839405, 3.5807128
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3978291, 2.4085720
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5699096, 2.5680718
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0403161, 2.0224736
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8214693, 2.8076730
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1021109, 3.1077790
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6149740, 2.6223900
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.2998199, 2.3041773

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1922462, upper bound: 1.1660351
time: 10.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1919565, upper bound: 1.1663199
time: 43.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3525562, 3.3555031
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9485826, 2.9565811
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5798550, 3.5847983
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3984089, 2.4079924
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5725913, 2.5653911
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0404935, 2.0222960
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8171453, 2.8119965
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1043110, 3.1055784
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6134310, 2.6239343
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3020363, 2.3019609

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1914739, upper bound: 1.1668042
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1911827, upper bound: 1.1670909
time: 12.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3538017, 3.3542571
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9496832, 2.9554796
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5813274, 3.5833259
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3974562, 2.4089453
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5724292, 2.5655532
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0388288, 2.0239606
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8199205, 2.8092213
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.1028032, 3.1070871
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6143446, 2.6230199
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3026009, 2.3013964

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 915

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1914639, upper bound: 1.1668162
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1911743, upper bound: 1.1671032
time: 12.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 36.67 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1922560, upper bound: 1.1660227
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1919605, upper bound: 1.1663072
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1922462, upper bound: 1.1660351
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1919565, upper bound: 1.1663199
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1914739, upper bound: 1.1668042
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1911827, upper bound: 1.1670909
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1914639, upper bound: 1.1668162
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 36.67
Output dim: 5, lower bound: -1.1911743, upper bound: 1.1671032

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3529596, 3.3543153
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9446735, 2.9441934
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5811834, 3.5817447
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3993368, 2.4080989
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5724673, 2.5706758
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0418208, 2.0206687
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8138347, 2.8048935
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.0967407, 3.1002517
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6137881, 2.6230054
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3008919, 2.3066316

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1922478, upper bound: 1.1640619
time: 10.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1902982, upper bound: 1.1660131
time: 12.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3530121, 3.3542628
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9457588, 2.9431086
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5820265, 3.5809021
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3992624, 2.4081733
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5728383, 2.5703058
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0418408, 2.0206487
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8131394, 2.8055882
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.0976000, 3.0993919
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6137614, 2.6230330
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3011446, 2.3063788

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1919523, upper bound: 1.1643506
time: 12.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1900017, upper bound: 1.1663006
time: 14.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3542051, 3.3530693
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9457760, 2.9430919
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5826569, 3.5802717
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3983831, 2.4090519
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5723052, 2.5708380
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0401556, 2.0223334
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8166089, 2.8021183
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.0952320, 3.1017599
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6147037, 2.6220911
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3014565, 2.3060670

Time for backsubstitution: 15.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1922381, upper bound: 1.1640749
time: 8.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1902881, upper bound: 1.1660276
time: 8.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.0934620, -5.3853927, -9.0934620, -5.3853927, -3.3542585, 3.3530169
1: -11.2401772, -7.5092487, -11.2401772, -7.5092487, -2.9468613, 2.9420071
2: -10.3444309, -6.3544011, -10.3444309, -6.3544011, -3.5834990, 3.5794291
3: -5.0488024, -2.3199015, -5.0488024, -2.3199015, -2.3983097, 2.4091260
4: -11.4109173, -8.3298731, -11.4109173, -8.3298731, -2.5726762, 2.5704679
5: 6.9647899, 9.4015284, 6.9647899, 9.4015284, -2.0401757, 2.0223136
6: -8.6112747, -5.0921683, -8.6112747, -5.0921683, -2.8159146, 2.8028131
7: -17.1788979, -13.3413029, -17.1788979, -13.3413029, -3.0960913, 3.1009007
8: -6.0857444, -3.1872163, -6.0857444, -3.1872163, -2.6146750, 2.6221187
9: -4.2306423, -1.7395773, -4.2306423, -1.7395773, -2.3017092, 2.3058143

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4636
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 137
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 4636

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1919483, upper bound: 1.1643618
time: 11.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1899984, upper bound: 1.1663132
time: 8.02 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 34.25 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1922478, upper bound: 1.1640619
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1902982, upper bound: 1.1660131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1919523, upper bound: 1.1643506
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1900017, upper bound: 1.1663006
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1922381, upper bound: 1.1640749
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1902881, upper bound: 1.1660276
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1919483, upper bound: 1.1643618
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 34.25
Output dim: 5, lower bound: -1.1899984, upper bound: 1.1663132
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 34.25
Output dim: 5, lower bound: -1.1914739, upper bound: 1.1668042
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 34.25
Output dim: 5, lower bound: -1.1911827, upper bound: 1.1670909
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 34.25
Output dim: 5, lower bound: -1.1914639, upper bound: 1.1668162
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 34.25
Output dim: 5, lower bound: -1.1911743, upper bound: 1.1671032
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.1325697898864746
rel_dist={5: [-1.1923778842711146, 1.1923772606866638]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2426.64 seconds
