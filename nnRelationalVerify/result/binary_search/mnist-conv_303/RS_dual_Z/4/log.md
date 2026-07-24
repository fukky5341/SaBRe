## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.81814518181
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.8957224, 11.2440281, 8.8957224, 11.2440281, -2.3483057, 2.3483057)
1: (-19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.8090019, 3.8090019)
2: (-3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.8001652, 2.8001652)
3: (-13.5667639, -10.0373669, -13.5667639, -10.0373669, -3.5293970, 3.5293970)
4: (-15.5752630, -11.9897022, -15.5752630, -11.9897022, -3.5616326, 3.5616322)
5: (-6.1198454, -3.7446783, -6.1198454, -3.7446783, -2.3524384, 2.3524384)
6: (-3.6156614, -1.3861105, -3.6156614, -1.3861105, -2.2295508, 2.2295508)
7: (-7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.8272722, 3.8272722)
8: (-2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.1913996, 2.1913996)
9: (-9.4011059, -6.0042830, -9.4011059, -6.0042830, -3.2635217, 3.2635217)

## BASE Result
execution time: IAR + LP analysis = 15.00 + 32.67 = 47.67 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.33 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.031045436859131
rel_dist={0: [-1.052260884305138, 1.052260719977161]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.8055601119995117
rel_dist={0: [-0.697324300994195, 0.6973245229667757]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.8807220458984375
rel_dist={0: [-0.8193859179682992, 0.8193858024795073]}

## Binary Search Result
Binary search time: 155.92 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_dual_Z) starts
Time budget: 3396.41 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=None

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9371681, upper bound: 0.9370326
time: 9.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9370303, upper bound: 0.9371684
time: 5.10 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.41
Output dim: 0, lower bound: -0.9371681, upper bound: 0.9370326
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.41
Output dim: 0, lower bound: -0.9370303, upper bound: 0.9371684

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9621630, 1.9595451
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2810431, 3.2788577
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2848487, 2.2813382
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7822070, 2.7788358
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6768713, 2.6937647
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7587118, 1.7582664
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9219785, 1.9184420
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7097607, 3.7140779
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0461645, 2.0409393
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5080819, 2.5034895

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9360053, upper bound: 0.9370263
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9371643, upper bound: 0.9358677
time: 7.25 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9595451, 1.9621632
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2788572, 3.2810426
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2813392, 2.2848480
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7788367, 2.7822061
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6937647, 2.6768703
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7582664, 1.7587118
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9184418, 1.9219785
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7140779, 3.7097611
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0409393, 2.0461645
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5034895, 2.5080819

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9358677, upper bound: 0.9371644
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9370264, upper bound: 0.9360053
time: 4.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 0, lower bound: -0.9360053, upper bound: 0.9370263
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 0, lower bound: -0.9371643, upper bound: 0.9358677
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 0, lower bound: -0.9358677, upper bound: 0.9371644
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.99
Output dim: 0, lower bound: -0.9370264, upper bound: 0.9360053

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9575362, 1.9561915
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2730227, 3.2743526
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2848635, 2.2813814
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7772789, 2.7766457
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6740522, 2.6874576
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7550495, 1.7553682
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9021292, 1.9027221
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6967649, 3.6976519
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0453596, 2.0403018
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5075464, 2.5032487

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9359933, upper bound: 0.9256578
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9246177, upper bound: 0.9370146
time: 6.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9588094, 1.9549179
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2765379, 3.2708373
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2848902, 2.2813542
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7800169, 2.7739086
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6705637, 2.6909461
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7558138, 1.7546039
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9062591, 1.8985925
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6933355, 3.7010818
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0455265, 2.0401349
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5078406, 2.5029545

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9371524, upper bound: 0.9244969
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257768, upper bound: 0.9358558
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9549179, 1.9588094
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2708368, 3.2765374
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2813540, 2.2848907
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7739086, 2.7800159
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6909466, 2.6705632
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7546041, 1.7558136
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8985925, 1.9062588
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7010822, 3.6933346
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0401344, 2.0455270
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5029540, 2.5078406

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9358557, upper bound: 0.9257791
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244966, upper bound: 0.9371528
time: 5.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9561915, 1.9575360
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2743521, 3.2730222
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2813807, 2.2848639
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7766457, 2.7772789
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6874580, 2.6740518
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7553685, 1.7550492
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9027219, 1.9021292
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6976519, 3.6967649
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0403023, 2.0453596
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5032487, 2.5075469

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9370144, upper bound: 0.9246176
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256554, upper bound: 0.9359938
time: 6.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9359933, upper bound: 0.9256578
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9246177, upper bound: 0.9370146
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9371524, upper bound: 0.9244969
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9257768, upper bound: 0.9358558
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9358557, upper bound: 0.9257791
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9244966, upper bound: 0.9371528
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9370144, upper bound: 0.9246176
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.05
Output dim: 0, lower bound: -0.9256554, upper bound: 0.9359938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9491043, 1.9368136
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2639589, 3.2534795
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2830334, 2.2771797
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7762766, 2.7762089
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6729565, 2.6869869
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7502601, 1.7532887
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8934002, 1.8826256
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6825113, 3.6914396
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0414038, 2.0385833
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5071602, 2.5023623

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9320686, upper bound: 0.9256552
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9359906, upper bound: 0.9217309
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9381580, 1.9477596
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2521486, 3.2652893
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2806625, 2.2795501
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7768421, 2.7756433
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6735811, 2.6863618
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7529700, 1.7505791
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8820324, 1.8939931
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6905527, 3.6833978
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0436411, 2.0363460
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5066605, 2.5028620

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9206930, upper bound: 0.9370142
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9246150, upper bound: 0.9330879
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9503775, 1.9355402
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2674742, 3.2499642
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2830601, 2.2771525
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7790146, 2.7734716
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6694679, 2.6904755
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7510245, 1.7525244
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8975296, 1.8784962
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6790810, 3.6948695
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0415707, 2.0384159
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5074544, 2.5020680

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9332278, upper bound: 0.9244963
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9371497, upper bound: 0.9205698
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9394312, 1.9464860
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2556639, 3.2617741
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2806892, 2.2795229
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7795792, 2.7729063
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6700926, 2.6898503
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7537344, 1.7498147
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8861623, 1.8898635
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6871233, 3.6868277
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0438080, 2.0361786
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5069542, 2.5025678

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9218523, upper bound: 0.9358531
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9257741, upper bound: 0.9319288
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9464860, 1.9394312
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2617741, 3.2556643
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2795238, 2.2806890
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7729063, 2.7795792
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6898499, 2.6700926
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7498147, 1.7537341
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8898635, 1.8861623
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6868277, 3.6871223
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0361786, 2.0438080
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5025678, 2.5069544

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9319288, upper bound: 0.9257765
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9358530, upper bound: 0.9218547
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9355402, 1.9503772
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2499647, 3.2674742
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2771530, 2.2830594
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7734718, 2.7790136
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6904755, 2.6694674
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7525246, 1.7510245
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8784962, 1.8975296
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6948700, 3.6790810
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0384159, 2.0415707
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5020680, 2.5074544

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9205697, upper bound: 0.9371501
time: 8.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9244939, upper bound: 0.9332281
time: 8.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9477596, 1.9381580
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2652893, 3.2521491
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2795506, 2.2806623
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7756433, 2.7768421
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6863613, 2.6735811
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7505791, 1.7529700
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8939929, 1.8820329
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6833982, 3.6905527
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0363464, 2.0436411
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5028620, 2.5066605

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9330877, upper bound: 0.9246150
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9370118, upper bound: 0.9206930
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9368134, 1.9491041
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2534800, 3.2639589
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2771797, 2.2830327
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7762089, 2.7762766
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6869869, 2.6729560
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7532890, 1.7502601
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8826256, 1.8934002
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6914396, 3.6825113
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0385838, 2.0414038
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5023623, 2.5071602

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9217285, upper bound: 0.9359930
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9256528, upper bound: 0.9320710
time: 4.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9320686, upper bound: 0.9256552
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9359906, upper bound: 0.9217309
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9206930, upper bound: 0.9370142
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9246150, upper bound: 0.9330879
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9332278, upper bound: 0.9244963
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9371497, upper bound: 0.9205698
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9218523, upper bound: 0.9358531
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9257741, upper bound: 0.9319288
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9319288, upper bound: 0.9257765
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9358530, upper bound: 0.9218547
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9205697, upper bound: 0.9371501
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9244939, upper bound: 0.9332281
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9330877, upper bound: 0.9246150
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9370118, upper bound: 0.9206930
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9217285, upper bound: 0.9359930
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.91
Output dim: 0, lower bound: -0.9256528, upper bound: 0.9320710

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9361124, 1.9265285
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2451572, 3.2297525
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2731667, 2.2716322
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7569704, 2.7609203
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6651101, 2.6725898
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7495079, 1.7518337
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8968987, 1.8846424
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6667099, 3.6714821
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0334406, 2.0322800
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4823170, 2.4840813

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9311039, upper bound: 0.9250015
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9311050, upper bound: 0.9207686
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9388185, 1.9238219
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2402325, 3.2346778
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2774830, 2.2673132
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7609882, 2.7569022
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6585584, 2.6791372
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7488050, 1.7525365
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8954167, 1.8861246
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6625538, 3.6756387
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0351000, 2.0306201
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4888763, 2.4775193

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9311082, upper bound: 0.9207665
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9353361, upper bound: 0.9207663
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9251661, 1.9374745
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2333469, 3.2415624
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2707958, 2.2740026
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7575359, 2.7603550
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6657357, 2.6719646
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7522173, 1.7491238
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8855309, 1.8960099
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6747513, 3.6634402
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0356779, 2.0300426
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4818172, 2.4845810

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9197267, upper bound: 0.9363576
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9197274, upper bound: 0.9321278
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9278727, 1.9347677
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2284222, 3.2464876
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2751131, 2.2696836
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7615538, 2.7563369
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6591840, 2.6785116
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7515144, 1.7498269
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8840499, 1.8974922
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6705952, 3.6675968
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0373373, 2.0283833
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4883766, 2.4780190

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9197319, upper bound: 0.9321244
time: 9.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9239607, upper bound: 0.9321242
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9373856, 1.9252553
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2486734, 3.2262373
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2731934, 2.2716045
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7597075, 2.7581832
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6616216, 2.6760783
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7502728, 1.7510693
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9010301, 1.8805130
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6632805, 3.6749125
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0336075, 2.0321126
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4826112, 2.4837861

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9322656, upper bound: 0.9238428
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9322666, upper bound: 0.9196096
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9400911, 1.9225483
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2437468, 3.2311616
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2775116, 2.2672861
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7637253, 2.7541652
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6550698, 2.6826243
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7495689, 1.7517719
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8995461, 1.8819933
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6591234, 3.6790681
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0352678, 2.0304527
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4891715, 2.4772248

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9322686, upper bound: 0.9196019
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9364940, upper bound: 0.9196016
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9264393, 1.9362013
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2368641, 3.2380471
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2708225, 2.2739749
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7602730, 2.7576180
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6622472, 2.6754532
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7529821, 1.7483594
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8896623, 1.8918803
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6713219, 3.6668706
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0358448, 2.0298753
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4821115, 2.4842858

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9208896, upper bound: 0.9352000
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9208906, upper bound: 0.9309663
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9291453, 1.9334941
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2319374, 3.2429709
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2751408, 2.2696564
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7642908, 2.7535999
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6556954, 2.6819992
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7522793, 1.7490623
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8881793, 1.8933609
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6671658, 3.6710262
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0375042, 2.0282159
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4886718, 2.4777246

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9208926, upper bound: 0.9309610
time: 7.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9251186, upper bound: 0.9309605
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9334941, 1.9291453
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2429714, 3.2319374
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2696571, 2.2751405
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7536001, 2.7642906
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6819997, 2.6556959
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7490625, 1.7522790
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8933606, 1.8881791
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6710262, 3.6671653
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0282154, 2.0375047
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4777246, 2.4886718

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9309604, upper bound: 0.9251179
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9309608, upper bound: 0.9208918
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9362011, 1.9264395
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2380466, 3.2368636
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2739754, 2.2708225
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7576180, 2.7602725
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6754537, 2.6622472
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7483597, 1.7529821
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8918805, 1.8896625
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6668701, 3.6713219
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0298748, 2.0358453
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4842858, 2.4821115

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9309663, upper bound: 0.9208904
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9352001, upper bound: 0.9208907
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9225483, 1.9400911
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2311611, 3.2437472
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2672863, 2.2775109
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7541656, 2.7637253
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6826253, 2.6550703
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7517719, 1.7495692
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8819938, 1.8995464
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6790676, 3.6591234
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0304527, 2.0352678
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4772248, 2.4891715

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9196015, upper bound: 0.9364943
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9196020, upper bound: 0.9322688
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9252553, 1.9373856
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2262373, 3.2486730
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2716045, 2.2731929
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7581835, 2.7597075
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6760793, 2.6616220
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7510691, 1.7502725
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8805127, 1.9010301
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6749125, 3.6632800
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0321121, 2.0336080
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4837861, 2.4826112

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9196072, upper bound: 0.9322662
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9238427, upper bound: 0.9322658
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9347677, 1.9278724
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2464876, 3.2284222
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2696838, 2.2751129
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7563372, 2.7615535
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6785111, 2.6591845
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7498264, 1.7515147
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8974919, 1.8840497
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6675968, 3.6705956
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0283833, 2.0373373
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4780188, 2.4883766

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9321240, upper bound: 0.9239601
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9321247, upper bound: 0.9197315
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9374743, 1.9251661
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2415619, 3.2333469
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2740030, 2.2707958
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7603550, 2.7575355
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6719651, 2.6657352
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7491236, 1.7522175
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8960099, 1.8855312
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6634407, 3.6747513
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0300426, 2.0356779
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4845810, 2.4818172

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9321278, upper bound: 0.9197271
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9363573, upper bound: 0.9197272
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9238219, 1.9388185
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2346783, 3.2402320
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2673130, 2.2774835
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7569027, 2.7609882
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6791368, 2.6585588
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7525368, 1.7488048
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8861251, 1.8954170
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6756382, 3.6625538
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0306206, 2.0351000
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4775190, 2.4888766

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207653, upper bound: 0.9353384
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9207671, upper bound: 0.9311088
time: 4.74 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9311039, upper bound: 0.9250015
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9311050, upper bound: 0.9207686
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9311082, upper bound: 0.9207665
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9353361, upper bound: 0.9207663
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9197267, upper bound: 0.9363576
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9197274, upper bound: 0.9321278
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9197319, upper bound: 0.9321244
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9239607, upper bound: 0.9321242
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9322656, upper bound: 0.9238428
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9322666, upper bound: 0.9196096
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9322686, upper bound: 0.9196019
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9364940, upper bound: 0.9196016
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9208896, upper bound: 0.9352000
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9208906, upper bound: 0.9309663
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9208926, upper bound: 0.9309610
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9251186, upper bound: 0.9309605
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9309604, upper bound: 0.9251179
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9309608, upper bound: 0.9208918
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9309663, upper bound: 0.9208904
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9352001, upper bound: 0.9208907
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9196015, upper bound: 0.9364943
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9196020, upper bound: 0.9322688
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9196072, upper bound: 0.9322662
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9238427, upper bound: 0.9322658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9321240, upper bound: 0.9239601
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9321247, upper bound: 0.9197315
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9321278, upper bound: 0.9197271
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9363573, upper bound: 0.9197272
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9207653, upper bound: 0.9353384
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.94
Output dim: 0, lower bound: -0.9207671, upper bound: 0.9311088
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.94
Output dim: 0, lower bound: -0.9256528, upper bound: 0.9320710
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.955883502960205
rel_dist={0: [-0.9371681487262702, 0.9371704841879342]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193859, upper bound: 0.8193882
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193859, upper bound: 0.8193882
time: 8.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.28
Output dim: 0, lower bound: -0.8193859, upper bound: 0.8193882
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.28
Output dim: 0, lower bound: -0.8193859, upper bound: 0.8193882

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8864775, 1.8843837
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1593199, 3.1575713
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2044311, 2.2016239
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6741710, 2.6714754
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5616713, 2.5751867
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6740465, 1.6736901
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8582463, 1.8554173
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5791349, 3.5825882
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9832258, 1.9790459
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4017105, 2.3980367

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8184121, upper bound: 0.8193822
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193826, upper bound: 0.8184123
time: 8.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8843832, 1.8864779
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1575708, 3.1593194
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2016234, 2.2044318
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6714759, 2.6741717
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5751867, 2.5616713
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6736903, 1.6740465
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8554173, 1.8582463
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5825872, 3.5791345
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9790459, 1.9832258
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3980365, 2.4017107

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8184121, upper bound: 0.8193823
time: 9.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193826, upper bound: 0.8184120
time: 8.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 32.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 32.47
Output dim: 0, lower bound: -0.8184121, upper bound: 0.8193822
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 32.47
Output dim: 0, lower bound: -0.8193826, upper bound: 0.8184123
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 32.47
Output dim: 0, lower bound: -0.8184121, upper bound: 0.8193823
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 32.47
Output dim: 0, lower bound: -0.8193826, upper bound: 0.8184120

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8818507, 1.8807750
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1512995, 3.1523633
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2044477, 2.2016616
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6692448, 2.6687379
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5581551, 2.5688796
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6703837, 1.6706390
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8383970, 1.8388715
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5654526, 3.5661616
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9824209, 1.9783750
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4011750, 2.3977370

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8184030, upper bound: 0.8100723
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8091192, upper bound: 0.8193758
time: 8.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8828692, 1.8797560
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1541109, 3.1495514
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2044687, 2.2016401
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6714344, 2.6665483
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5553646, 2.5716705
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6709955, 1.6700275
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8417006, 1.8355677
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5627089, 3.5689058
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9825554, 1.9782410
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4014106, 2.3975015

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193735, upper bound: 0.8091007
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100962, upper bound: 0.8184038
time: 7.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8797565, 1.8828692
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1495514, 3.1541109
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2016401, 2.2044692
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6665478, 2.6714342
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5716705, 2.5553641
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6700275, 1.6709952
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8355680, 1.8417006
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5689058, 3.5627084
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9782410, 1.9825549
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3975015, 2.4014106

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8184030, upper bound: 0.8100962
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8091006, upper bound: 0.8193736
time: 7.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8807750, 1.8818502
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1523638, 3.1512990
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2016611, 2.2044477
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6687384, 2.6692445
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5688801, 2.5581555
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6706393, 1.6703839
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8388715, 1.8383970
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5661621, 3.5654526
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9783745, 1.9824209
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3977370, 2.4011755

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193735, upper bound: 0.8091193
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8100718, upper bound: 0.8184028
time: 9.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 30.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8184030, upper bound: 0.8100723
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8091192, upper bound: 0.8193758
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8193735, upper bound: 0.8091007
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8100962, upper bound: 0.8184038
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8184030, upper bound: 0.8100962
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8091006, upper bound: 0.8193736
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8193735, upper bound: 0.8091193
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 30.42
Output dim: 0, lower bound: -0.8100718, upper bound: 0.8184028

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8712292, 1.8613968
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1398735, 3.1314902
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2021427, 2.1974599
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6682425, 2.6681876
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5570593, 2.5682840
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6655948, 1.6680176
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8273945, 1.8187749
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5511990, 3.5583410
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9784651, 1.9762087
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4006886, 2.3968506

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152330, upper bound: 0.8100692
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8184008, upper bound: 0.8069129
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8624725, 1.8701539
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1304255, 3.1409378
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2002468, 2.1993563
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6686945, 2.6677356
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5575600, 2.5677838
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6677625, 1.6658497
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8183002, 1.8278689
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5576324, 3.5519075
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9802551, 1.9744186
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4002891, 2.3972507

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8059400, upper bound: 0.8193737
time: 7.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8091171, upper bound: 0.8162117
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8722477, 1.8603783
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1426859, 3.1286783
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2021637, 2.1974382
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6704321, 2.6659982
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5542688, 2.5710750
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6662061, 1.6674058
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8306980, 1.8154712
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5484543, 3.5610852
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9785995, 1.9760747
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4009242, 2.3966153

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8162111, upper bound: 0.8090986
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193714, upper bound: 0.8059167
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8634911, 1.8691349
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1332378, 3.1381259
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2002678, 2.1993346
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6708841, 2.6655459
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5547686, 2.5705748
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6683738, 1.6652381
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8216038, 1.8245652
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5548878, 3.5546517
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9803886, 1.9742851
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4005241, 2.3970151

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069325, upper bound: 0.8184011
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100942, upper bound: 0.8152337
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8691349, 1.8634911
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1381254, 3.1332378
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1993351, 2.2002676
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6655464, 2.6708841
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5705748, 2.5547690
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6652381, 1.6683738
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8245649, 1.8216043
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5546522, 3.5548878
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9742851, 1.9803886
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3970151, 2.4005244

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152330, upper bound: 0.8100944
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8184008, upper bound: 0.8069323
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8603783, 1.8722479
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1286783, 3.1426859
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1974392, 2.2021637
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6659985, 2.6704319
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5710754, 2.5542688
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6674058, 1.6662061
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8154712, 1.8306980
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5610857, 3.5484543
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9760752, 1.9785986
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3966150, 2.4009242

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8059169, upper bound: 0.8193713
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8090985, upper bound: 0.8162115
time: 5.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8701539, 1.8624725
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1409378, 3.1304259
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1993561, 2.2002461
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6677361, 2.6686945
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5677843, 2.5575600
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6658499, 1.6677623
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8278685, 1.8183005
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5519075, 3.5576320
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9744186, 1.9802547
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3972507, 2.4002893

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8162111, upper bound: 0.8091172
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193714, upper bound: 0.8059399
time: 7.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8613968, 1.8712294
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1314898, 3.1398735
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1974602, 2.2021425
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6681881, 2.6682422
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5682840, 2.5570593
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6680176, 1.6655946
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8187747, 1.8273945
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5583410, 3.5511985
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9762087, 1.9784651
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3968506, 2.4006891

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069131, upper bound: 0.8184032
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8100697, upper bound: 0.8152330
time: 5.45 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8152330, upper bound: 0.8100692
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8184008, upper bound: 0.8069129
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8059400, upper bound: 0.8193737
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8091171, upper bound: 0.8162117
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8162111, upper bound: 0.8090986
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8193714, upper bound: 0.8059167
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8069325, upper bound: 0.8184011
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8100942, upper bound: 0.8152337
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8152330, upper bound: 0.8100944
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8184008, upper bound: 0.8069323
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8059169, upper bound: 0.8193713
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8090985, upper bound: 0.8162115
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8162111, upper bound: 0.8091172
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8193714, upper bound: 0.8059399
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8069131, upper bound: 0.8184032
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.84
Output dim: 0, lower bound: -0.8100697, upper bound: 0.8152330

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8604026, 1.8484054
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1161470, 3.1117034
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1957302, 2.1875935
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6521502, 2.6488814
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5426621, 2.5591249
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6641397, 1.6671247
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8294110, 1.8219774
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5312414, 3.5417089
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9718294, 1.9682455
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3810935, 2.3720076

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8143568, upper bound: 0.8060304
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8177773, upper bound: 0.8060270
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8494806, 1.8593273
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1106386, 3.1172109
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1903801, 2.1929452
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6493883, 2.6516435
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5484033, 2.5533867
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6668692, 1.6643946
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8215032, 1.8298857
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5409994, 3.5319505
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9722919, 1.9677835
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3754458, 2.3776569

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8050648, upper bound: 0.8187409
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8050652, upper bound: 0.8153319
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8614206, 1.8473864
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1189585, 3.1088905
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1957521, 2.1875720
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6543398, 2.6466918
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5398717, 2.5619144
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6647511, 1.6665130
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8327146, 1.8186724
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5284967, 3.5444527
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9719639, 1.9681120
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3813291, 2.3717721

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8153323, upper bound: 0.8050445
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187410, upper bound: 0.8050439
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8504992, 1.8583088
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1134520, 3.1143990
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1904011, 2.1929228
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6515779, 2.6494539
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5456128, 2.5561776
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6674814, 1.6637831
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8248076, 1.8265820
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5382557, 3.5346947
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9724255, 1.9676495
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3756814, 2.3774209

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8060449, upper bound: 0.8177768
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8060454, upper bound: 0.8143562
time: 5.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8583088, 1.8504994
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1143990, 3.1134520
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1929226, 2.1904011
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6494541, 2.6515775
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5561776, 2.5456128
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6637831, 1.6674812
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8265824, 1.8248079
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5346947, 3.5382557
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9676495, 1.9724255
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3774209, 2.3756814

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8143568, upper bound: 0.8060478
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8177773, upper bound: 0.8060443
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8473864, 1.8614206
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1088905, 3.1189590
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1875725, 2.1957517
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6466923, 2.6543398
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5619149, 2.5398712
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6665125, 1.6647508
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8186727, 1.8327148
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5444527, 3.5284967
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9681120, 1.9719634
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3717723, 2.3813295

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8050442, upper bound: 0.8187412
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8050446, upper bound: 0.8153346
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8593273, 1.8494809
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1172104, 3.1106391
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1929455, 2.1903796
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6516438, 2.6493881
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5533872, 2.5484037
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6643944, 1.6668694
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8298860, 1.8215029
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5319500, 3.5409994
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9677830, 1.9722919
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3776569, 2.3754461

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8153323, upper bound: 0.8050652
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187410, upper bound: 0.8050647
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8484054, 1.8604026
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1117039, 3.1161466
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1875935, 2.1957297
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6488810, 2.6521502
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5591245, 2.5426626
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6671247, 1.6641395
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8219771, 1.8294113
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5417089, 3.5312414
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9682455, 1.9718294
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3720074, 2.3810935

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8060274, upper bound: 0.8177778
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8060280, upper bound: 0.8143564
time: 5.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8143568, upper bound: 0.8060304
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8177773, upper bound: 0.8060270
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8050648, upper bound: 0.8187409
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8050652, upper bound: 0.8153319
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8153323, upper bound: 0.8050445
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8187410, upper bound: 0.8050439
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8060449, upper bound: 0.8177768
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8060454, upper bound: 0.8143562
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8143568, upper bound: 0.8060478
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8177773, upper bound: 0.8060443
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8050442, upper bound: 0.8187412
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8050446, upper bound: 0.8153346
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8153323, upper bound: 0.8050652
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8187410, upper bound: 0.8050647
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8060274, upper bound: 0.8177778
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.52
Output dim: 0, lower bound: -0.8060280, upper bound: 0.8143564

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8487525, 1.8589272
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1106358, 3.1187048
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1903696, 2.1945977
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6445456, 2.6489921
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5503321, 2.5533748
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6668634, 1.6654840
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8219104, 1.8298829
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5407019, 3.5314069
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9726362, 1.9677820
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3754425, 2.3782184

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8050631, upper bound: 0.8181966
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8045196, upper bound: 0.8187398
time: 4.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8610206, 1.8466580
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1204529, 3.1088877
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1974039, 2.1875625
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6516886, 2.6418498
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5398588, 2.5638418
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6658406, 1.6665070
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8327127, 1.8190801
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5279531, 3.5441551
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9719629, 1.9684553
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3818908, 2.3717687

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187393, upper bound: 0.8045004
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8181966, upper bound: 0.8050428
time: 5.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8466582, 1.8610208
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1088877, 3.1204524
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1875620, 2.1974039
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6418495, 2.6516883
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5638418, 2.5398593
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6665072, 1.6658404
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8190808, 1.8327124
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5441542, 3.5279536
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9684553, 1.9719620
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3717690, 2.3818908

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8050425, upper bound: 0.8181964
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8045014, upper bound: 0.8187390
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8589272, 1.8487525
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1187048, 3.1106358
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1945982, 2.1903703
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6489916, 2.6445460
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5533743, 2.5503316
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6654844, 1.6668634
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8298831, 1.8219106
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5314074, 3.5407019
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9677820, 1.9726367
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3782187, 2.3754427

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187393, upper bound: 0.8045220
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8181966, upper bound: 0.8050655
time: 5.53 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 25.36 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8050631, upper bound: 0.8181966
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8045196, upper bound: 0.8187398
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8187393, upper bound: 0.8045004
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8181966, upper bound: 0.8050428
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8050425, upper bound: 0.8181964
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8045014, upper bound: 0.8187390
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8187393, upper bound: 0.8045220
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.36
Output dim: 0, lower bound: -0.8181966, upper bound: 0.8050655

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8536115, 1.8623729
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.0945396, 3.1052861
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1964836, 2.2018034
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6137924, 2.6233518
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5441370, 2.5430732
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6668134, 1.6663539
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8174672, 1.8239844
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5322943, 3.5244026
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9697962, 1.9654136
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3742738, 2.3757858

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8037363, upper bound: 0.8076175
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8037341, upper bound: 0.8167311
time: 7.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8521976, 1.8637867
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.0972176, 3.1026092
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1975765, 2.2007103
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6189060, 2.6182382
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5400305, 2.5471797
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6677337, 1.6654339
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8160119, 1.8254399
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5336971, 3.5229993
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9702682, 1.9649405
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3730102, 2.3770494

Time for backsubstitution: 14.35 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.8807220458984375
rel_dist={0: [-0.8193859179682992, 0.8193858024795073]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1658.33 seconds
