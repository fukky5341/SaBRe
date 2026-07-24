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
execution time: IAR + LP analysis = 14.95 + 32.76 = 47.71 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.29 seconds, max iter: 100)

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
Binary search time: 155.99 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Relational Split (RS_random_Z) starts
Time budget: 3396.30 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=None

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9331632, upper bound: 0.9369521
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9369521, upper bound: 0.9331654
time: 5.23 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.50
Output dim: 0, lower bound: -0.9331632, upper bound: 0.9369521
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.50
Output dim: 0, lower bound: -0.9369521, upper bound: 0.9331654

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9558816, 1.9558821
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2743835, 3.2743845
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2756815, 2.2756824
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7950296, 2.7950301
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7552423, 2.7552409
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7604055, 1.7604048
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9147410, 1.9147432
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6899319, 3.6899300
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0414762, 2.0414762
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5253510, 2.5253506

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9331631, upper bound: 0.9368144
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9330261, upper bound: 0.9369520
time: 4.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9558821, 1.9558818
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2743855, 3.2743840
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2756824, 2.2756817
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7950296, 2.7950301
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7552404, 2.7552419
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7604046, 1.7604053
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9147429, 1.9147413
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6899300, 3.6899323
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0414762, 2.0414762
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5253501, 2.5253515

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 6137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9351481, upper bound: 0.9331569
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9369468, upper bound: 0.9313614
time: 5.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 0, lower bound: -0.9331631, upper bound: 0.9368144
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 0, lower bound: -0.9330261, upper bound: 0.9369520
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 0, lower bound: -0.9351481, upper bound: 0.9331569
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.60
Output dim: 0, lower bound: -0.9369468, upper bound: 0.9313614

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9621615, 1.9595437
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2810450, 3.2788610
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2848501, 2.2813411
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7822070, 2.7788360
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6768732, 2.6937666
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7587140, 1.7582684
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9219699, 1.9184351
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7097530, 3.7140670
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0461636, 2.0409384
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5080824, 2.5034888

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9331504, upper bound: 0.9254435
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9217769, upper bound: 0.9368025
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9595432, 1.9621618
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2788610, 3.2810459
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2813406, 2.2848508
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7788367, 2.7822063
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6937666, 2.6768723
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7582686, 1.7587137
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9184332, 1.9219716
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7140694, 3.7097502
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0409384, 2.0461631
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5034895, 2.5080812

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9330134, upper bound: 0.9255650
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216564, upper bound: 0.9369403
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9535427, 1.9540293
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2711697, 3.2716889
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2735710, 2.2731307
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7950945, 2.7951105
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7578039, 2.7573185
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7579052, 1.7572491
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9147425, 1.9148898
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6899748, 3.6900444
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0449753, 2.0457959
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5277104, 2.5272634

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9339853, upper bound: 0.9331533
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9351443, upper bound: 0.9319943
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9540296, 1.9535422
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2716894, 3.2711682
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2731314, 2.2735703
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7951097, 2.7950954
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7573185, 2.7578044
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7572486, 1.7579057
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9148917, 1.9147406
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6900425, 3.6899767
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0457954, 2.0449758
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5272622, 2.5277119

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9369448, upper bound: 0.9307165
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9362893, upper bound: 0.9313593
time: 5.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.83 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9331504, upper bound: 0.9254435
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9217769, upper bound: 0.9368025
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9330134, upper bound: 0.9255650
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9216564, upper bound: 0.9369403
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9339853, upper bound: 0.9331533
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9351443, upper bound: 0.9319943
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9369448, upper bound: 0.9307165
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.83
Output dim: 0, lower bound: -0.9362893, upper bound: 0.9313593

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9537292, 1.9401655
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2719831, 3.2579889
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2830191, 2.2771399
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7812047, 2.7783990
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6757765, 2.6932955
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7539241, 1.7561884
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9132404, 1.8983383
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6954985, 3.7078543
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0422068, 2.0392189
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5076952, 2.5026026

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 920

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9300678, upper bound: 0.9247941
time: 7.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9324947, upper bound: 0.9223654
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9427834, 1.9511113
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2601738, 3.2697988
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2806492, 2.2795103
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7817693, 2.7778337
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6764021, 2.6926703
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7566340, 1.7534785
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9018731, 1.9097056
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7035398, 3.6998124
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0444441, 2.0369821
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5071955, 2.5031023

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 5758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9217346, upper bound: 0.9367971
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9217705, upper bound: 0.9367666
time: 4.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9511108, 1.9427834
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2697983, 3.2601738
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2795095, 2.2806497
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7778344, 2.7817693
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6926718, 2.6764016
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7534788, 1.7566338
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9097037, 1.9018748
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6998148, 3.7035375
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0369816, 2.0444436
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5031033, 2.5071950

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329968, upper bound: 0.9255634
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9330115, upper bound: 0.9255476
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9401650, 1.9537294
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2579880, 3.2719836
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2771387, 2.2830200
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7783990, 2.7812040
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6932955, 2.6757760
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7561886, 1.7539241
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8983364, 1.9132423
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7078562, 3.6954956
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0392189, 2.0422068
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5026031, 2.5076947

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9198533, upper bound: 0.9369355
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216495, upper bound: 0.9351358
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9489155, 1.9506755
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2631502, 3.2671843
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2735882, 2.2731745
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7901678, 2.7929201
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7549858, 2.7510118
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7542429, 1.7543507
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8948936, 1.8991709
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6769800, 3.6736193
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0441713, 2.0451589
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5271769, 2.5270240

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9339430, upper bound: 0.9331476
time: 7.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9339789, upper bound: 0.9331118
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9501886, 1.9494021
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2666645, 3.2636695
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2736149, 2.2731473
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7929049, 2.7901828
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7514973, 2.7545009
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7550068, 1.7535863
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8990231, 1.8950412
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6735497, 3.6770496
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0443373, 2.0449915
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5274715, 2.5267296

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9351273, upper bound: 0.9319926
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9351426, upper bound: 0.9319774
time: 7.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9592414, 1.9569864
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2555904, 3.2584162
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2792435, 2.2810495
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7643557, 2.7707326
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7521501, 2.7475028
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7571976, 1.7590048
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9108133, 1.9088423
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6816359, 3.6833243
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0429540, 2.0427246
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5264087, 2.5252786

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9369328, upper bound: 0.9193303
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255576, upper bound: 0.9307030
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9574738, 1.9587541
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2589378, 3.2550702
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2806110, 2.2796829
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7707472, 2.7643404
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7470174, 2.7526360
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7583477, 1.7578549
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9089937, 1.9106619
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6833897, 3.6815701
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0435443, 2.0421338
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5248289, 2.5268579

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9362470, upper bound: 0.9313510
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9362829, upper bound: 0.9313149
time: 7.43 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9300678, upper bound: 0.9247941
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9324947, upper bound: 0.9223654
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9217346, upper bound: 0.9367971
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9217705, upper bound: 0.9367666
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9329968, upper bound: 0.9255634
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9330115, upper bound: 0.9255476
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9198533, upper bound: 0.9369355
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9216495, upper bound: 0.9351358
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9339430, upper bound: 0.9331476
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9339789, upper bound: 0.9331118
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9351273, upper bound: 0.9319926
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9351426, upper bound: 0.9319774
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9369328, upper bound: 0.9193303
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9255576, upper bound: 0.9307030
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9362470, upper bound: 0.9313510
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.04
Output dim: 0, lower bound: -0.9362829, upper bound: 0.9313149

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9530005, 1.9398470
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2719765, 3.2598524
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2820110, 2.2782011
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7763610, 2.7762940
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6742811, 2.6893768
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7539175, 1.7575505
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9137459, 1.8983364
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6952610, 3.7073107
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0426397, 2.0392184
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5064449, 2.5020568

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9282650, upper bound: 0.9247924
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9282685, upper bound: 0.9205528
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9534097, 1.9394369
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2738466, 3.2579823
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2840786, 2.2761323
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7790990, 2.7735560
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6718588, 2.6917953
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7552865, 1.7561817
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9132385, 1.8988423
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6949539, 3.7076178
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0422068, 2.0396495
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5071483, 2.5013516

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9306935, upper bound: 0.9223593
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9324859, upper bound: 0.9205640
time: 8.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9427829, 1.9511127
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2601719, 3.2698002
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2806487, 2.2795091
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7817712, 2.7778330
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6763983, 2.6926746
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7566340, 1.7534773
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9018755, 1.9097042
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7035379, 3.6998110
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0444469, 2.0369806
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5071979, 2.5031018

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9178098, upper bound: 0.9367944
time: 9.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9217320, upper bound: 0.9328698
time: 6.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9427834, 1.9511111
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2601738, 3.2697978
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2806487, 2.2795103
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7817683, 2.7778337
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6764021, 2.6926665
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7566340, 1.7534783
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9018717, 1.9097056
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7035398, 3.6998110
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0444431, 2.0369821
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5071955, 2.5031023

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9199665, upper bound: 0.9367603
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9217645, upper bound: 0.9349634
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9510126, 1.9417164
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2686939, 3.2600751
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2784204, 2.2682016
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7771187, 2.7736037
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6788092, 2.6751766
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7533085, 1.7566168
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9032922, 1.9013081
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6993780, 3.6986041
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0360250, 2.0443563
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5019093, 2.4934473

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9311938, upper bound: 0.9255583
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9329896, upper bound: 0.9237592
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9500442, 1.9426851
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2697001, 3.2590699
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2670612, 2.2795620
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7696686, 2.7810531
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6914549, 2.6625400
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7534621, 1.7564631
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9091358, 1.8954632
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6948814, 3.7031002
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0368958, 2.0434866
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4893556, 2.5060003

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 5758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9318490, upper bound: 0.9255441
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9330077, upper bound: 0.9243851
time: 5.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9378257, 1.9518766
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2547712, 3.2692876
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2750273, 2.2804694
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7784653, 2.7812853
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6958585, 2.6778531
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7536891, 1.7507682
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8983369, 1.9133923
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7079010, 3.6956077
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0427179, 2.0465250
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5049624, 2.5096056

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9198512, upper bound: 0.9362776
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9192160, upper bound: 0.9369351
time: 5.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9383125, 1.9513898
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2552919, 3.2687674
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2745886, 2.2809091
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7784805, 2.7812700
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6953740, 2.6783385
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7530329, 1.7514248
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8984861, 1.9132431
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.7079687, 3.6955400
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0435381, 2.0457048
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5045142, 2.5100539

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9177224, upper bound: 0.9351340
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9216469, upper bound: 0.9312116
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9489145, 1.9506764
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2631474, 3.2671857
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2735863, 2.2731733
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7901688, 2.7929192
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7549820, 2.7510142
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7542419, 1.7543495
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8948951, 1.8991694
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6769800, 3.6736183
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0441732, 2.0451574
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5271778, 2.5270228

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9300183, upper bound: 0.9331459
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9339403, upper bound: 0.9292247
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9489155, 1.9506748
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2631502, 3.2671828
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2735863, 2.2731745
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7901669, 2.7929201
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7549858, 2.7510090
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7542429, 1.7543504
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8948922, 1.8991709
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6769800, 3.6736193
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0441694, 2.0451589
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5271754, 2.5270240

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9309058, upper bound: 0.9324536
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9333312, upper bound: 0.9300266
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9500909, 1.9483354
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2655602, 3.2635698
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2725291, 2.2607000
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7921877, 2.7820163
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7376337, 2.7532778
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7548361, 1.7535694
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8926110, 1.8944721
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6731119, 3.6721168
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0433836, 2.0449080
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5262785, 2.5129824

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9351153, upper bound: 0.9206059
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9237401, upper bound: 0.9319794
time: 4.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9491220, 1.9493039
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2665653, 3.2625637
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2611670, 2.2720590
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7847385, 2.7894659
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7502708, 2.7406373
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7549901, 1.7534156
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8984542, 1.8886287
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6686163, 3.6766129
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0442553, 2.0440369
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5137239, 2.5255361

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9314777, upper bound: 0.9236234
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9267990, upper bound: 0.9283043
time: 5.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9508100, 1.9376092
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2465267, 3.2375422
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2774129, 2.2768478
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7633533, 2.7702956
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7510533, 2.7470317
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7524083, 1.7569253
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9020839, 1.8887458
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6673822, 3.6771121
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0389972, 2.0410056
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5260215, 2.5243921

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9330080, upper bound: 0.9193275
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9369302, upper bound: 0.9154052
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9398637, 1.9485550
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2347164, 3.2493515
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2750421, 2.2792182
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7639189, 2.7697303
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7516789, 2.7464066
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7551177, 1.7542155
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8907166, 1.9001131
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6754236, 3.6690698
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0412345, 2.0387683
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5255218, 2.5248919

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255153, upper bound: 0.9306970
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9255512, upper bound: 0.9306611
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.9574738, 1.9587553
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.2589359, 3.2550716
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2806101, 2.2796819
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.7707491, 2.7643394
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.7470136, 2.7526379
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.7583477, 1.7578537
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.9089952, 1.9106607
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.6833897, 3.6815691
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -2.0435467, 2.0421314
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.5248303, 2.5268574

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 6137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9362464, upper bound: 0.9313493
time: 10.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9362465, upper bound: 0.9313147
time: 5.10 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 30.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9282650, upper bound: 0.9247924
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9282685, upper bound: 0.9205528
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9306935, upper bound: 0.9223593
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9324859, upper bound: 0.9205640
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9178098, upper bound: 0.9367944
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9217320, upper bound: 0.9328698
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9199665, upper bound: 0.9367603
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9217645, upper bound: 0.9349634
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9311938, upper bound: 0.9255583
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9329896, upper bound: 0.9237592
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9318490, upper bound: 0.9255441
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9330077, upper bound: 0.9243851
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9198512, upper bound: 0.9362776
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9192160, upper bound: 0.9369351
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9177224, upper bound: 0.9351340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9216469, upper bound: 0.9312116
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9300183, upper bound: 0.9331459
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9339403, upper bound: 0.9292247
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9309058, upper bound: 0.9324536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9333312, upper bound: 0.9300266
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9351153, upper bound: 0.9206059
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9237401, upper bound: 0.9319794
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9314777, upper bound: 0.9236234
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9267990, upper bound: 0.9283043
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9330080, upper bound: 0.9193275
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9369302, upper bound: 0.9154052
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9255153, upper bound: 0.9306970
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9255512, upper bound: 0.9306611
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9362464, upper bound: 0.9313493
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 30.12
Output dim: 0, lower bound: -0.9362465, upper bound: 0.9313147
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 30.12
Output dim: 0, lower bound: -0.9362829, upper bound: 0.9313149
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.955883502960205
rel_dist={0: [-0.9371681487262702, 0.9371704841879342]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5758
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 920

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5758

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8162234, upper bound: 0.8193838
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8193839, upper bound: 0.8162233
time: 5.24 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.60
Output dim: 0, lower bound: -0.8162234, upper bound: 0.8193838
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.60
Output dim: 0, lower bound: -0.8193839, upper bound: 0.8162233

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8677297, 1.8698955
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1333113, 3.1293716
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1861000, 2.1895549
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6683645, 2.6715794
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6309204, 2.6256790
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6749344, 1.6743722
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8549204, 1.8537364
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5426826, 3.5393577
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9716196, 1.9729481
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3950586, 2.4003079

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8161796, upper bound: 0.8193759
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8162163, upper bound: 0.8193416
time: 5.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8698955, 1.8677297
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1293716, 3.1333113
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1895542, 2.1861000
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6715794, 2.6683650
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6256790, 2.6309204
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6743722, 1.6749344
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8537369, 1.8549206
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5393581, 3.5426831
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9729481, 1.9716201
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4003077, 2.3950589

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 6137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8153446, upper bound: 0.8153413
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187531, upper bound: 0.8153421
time: 8.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.94
Output dim: 0, lower bound: -0.8161796, upper bound: 0.8193759
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.94
Output dim: 0, lower bound: -0.8162163, upper bound: 0.8193416
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 27.94
Output dim: 0, lower bound: -0.8153446, upper bound: 0.8153413
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.94
Output dim: 0, lower bound: -0.8187531, upper bound: 0.8153421

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8677287, 1.8698955
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1333103, 3.1293726
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1860991, 2.1895540
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6683655, 2.6715782
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6309175, 2.6256804
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6749339, 1.6743710
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8549213, 1.8537350
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5426817, 3.5393558
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9716234, 1.9729471
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3950596, 2.4003069

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 920

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8152963, upper bound: 0.8187423
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152971, upper bound: 0.8153366
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8677297, 1.8698945
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1333113, 3.1293707
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1860991, 2.1895549
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6683645, 2.6715794
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6309204, 2.6256762
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6749344, 1.6743717
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8549194, 1.8537364
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5426826, 3.5393567
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9716196, 1.9729481
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3950577, 2.4003079

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 920

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8162072, upper bound: 0.8100534
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8069295, upper bound: 0.8193325
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8694944, 1.8670006
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1308613, 3.1333041
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1912050, 2.1860905
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6689262, 2.6635218
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6256666, 2.6328464
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6754603, 1.6749275
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8537335, 1.8553283
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5388155, 3.5423856
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9729481, 1.9719644
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4008684, 2.3950551

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187514, upper bound: 0.8147970
time: 8.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8182088, upper bound: 0.8153398
time: 15.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 38.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.95
Output dim: 0, lower bound: -0.8152963, upper bound: 0.8187423
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 38.95
Output dim: 0, lower bound: -0.8152971, upper bound: 0.8153366
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 38.95
Output dim: 0, lower bound: -0.8162072, upper bound: 0.8100534
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.95
Output dim: 0, lower bound: -0.8069295, upper bound: 0.8193325
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.95
Output dim: 0, lower bound: -0.8187514, upper bound: 0.8147970
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.95
Output dim: 0, lower bound: -0.8182088, upper bound: 0.8153398

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8670001, 1.8694949
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1333036, 3.1308622
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1860905, 2.1912045
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6635218, 2.6689253
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6328440, 2.6256676
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6749272, 1.6754594
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8553290, 1.8537321
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5423841, 3.5388131
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9719677, 1.9729466
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3950558, 2.4008679

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8152963, upper bound: 0.8187392
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8152963, upper bound: 0.8187423
time: 6.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8483520, 1.8592739
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1124382, 3.1179457
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1818981, 2.1872494
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6678143, 2.6705775
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6303244, 2.6245794
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6723123, 1.6695821
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8348227, 1.8427336
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5348625, 3.5251026
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9694538, 1.9689927
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3941712, 2.3998215

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8055674, upper bound: 0.8193283
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8069269, upper bound: 0.8179689
time: 4.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8743534, 1.8704457
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1147623, 3.1198821
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1973181, 2.1932967
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6381721, 2.6378808
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6194715, 2.6225457
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6754100, 1.6757970
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8492908, 1.8494306
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5304089, 3.5353823
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9701056, 1.9695954
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3996987, 2.3926215

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8177847, upper bound: 0.8147944
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187484, upper bound: 0.8138176
time: 5.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8729396, 1.8718600
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1174402, 3.1172051
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1984110, 2.1922035
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6432858, 2.6327672
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6153650, 2.6266522
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6763299, 1.6748769
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8478355, 1.8508861
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5318117, 3.5339789
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9705787, 1.9691224
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3984351, 2.3938849

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8182087, upper bound: 0.8153393
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8182087, upper bound: 0.8153391
time: 7.50 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8152963, upper bound: 0.8187392
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8152963, upper bound: 0.8187423
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8055674, upper bound: 0.8193283
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8069269, upper bound: 0.8179689
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8177847, upper bound: 0.8147944
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8187484, upper bound: 0.8138176
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8182087, upper bound: 0.8153393
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.66
Output dim: 0, lower bound: -0.8182087, upper bound: 0.8153391

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8727560, 1.8731565
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1395273, 3.1353383
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1945543, 2.1968606
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6500239, 2.6527317
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5544462, 2.5607882
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6731467, 1.6733227
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8618569, 1.8574286
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5622053, 3.5620880
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9756103, 1.9724083
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3768625, 2.3790007

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 920

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8152947, upper bound: 0.8187395
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8152946, upper bound: 0.8186988
time: 6.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8706617, 1.8752503
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1377792, 3.1370859
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1917467, 2.1996672
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6473289, 2.6554279
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5679579, 2.5472703
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6727905, 1.6736791
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8590260, 1.8602579
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5656586, 3.5586338
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9714284, 1.9765882
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3731885, 2.3826730

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8139347, upper bound: 0.8187377
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152945, upper bound: 0.8173783
time: 7.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8460112, 1.8573234
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1092224, 3.1151462
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1797862, 2.1847856
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6678801, 2.6706553
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6327896, 2.6266565
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6696820, 1.6664264
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8348227, 1.8428531
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5349207, 3.5252156
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9729524, 1.9731483
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3964410, 2.4017324

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8046793, upper bound: 0.8186919
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8046801, upper bound: 0.8152911
time: 6.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8707457, 1.8658190
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1095572, 3.1118641
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1973605, 2.1933153
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6354342, 2.6329534
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6131592, 2.6190257
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6723595, 1.6721346
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8327475, 1.8295834
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5139818, 3.5216994
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9694352, 1.9687920
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3994017, 2.3920882

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 920

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8173870, upper bound: 0.8138134
time: 9.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8187466, upper bound: 0.8124532
time: 7.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8786950, 1.8755212
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1236649, 3.1216831
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2068748, 2.1978598
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6297870, 2.6165724
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5369682, 2.5617652
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6745493, 1.6727407
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8543601, 1.8545811
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5516310, 3.5572519
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9742208, 1.9685855
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3802414, 2.3720186

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152485, upper bound: 0.8151814
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8180507, upper bound: 0.8123558
time: 6.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8766012, 1.8776159
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1219168, 3.1234312
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.2040672, 2.2006676
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6270909, 2.6192687
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5504837, 2.5482545
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6741936, 1.6730971
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8515306, 1.8574114
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5550842, 3.5537982
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9700408, 1.9727664
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3765688, 2.3756926

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152485, upper bound: 0.8151812
time: 5.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8180507, upper bound: 0.8123560
time: 8.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8152947, upper bound: 0.8187395
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8152946, upper bound: 0.8186988
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8139347, upper bound: 0.8187377
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8152945, upper bound: 0.8173783
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8046793, upper bound: 0.8186919
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8046801, upper bound: 0.8152911
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8173870, upper bound: 0.8138134
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8187466, upper bound: 0.8124532
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8152485, upper bound: 0.8151814
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8180507, upper bound: 0.8123558
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8152485, upper bound: 0.8151812
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.49
Output dim: 0, lower bound: -0.8180507, upper bound: 0.8123560

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8724642, 1.8720894
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1384230, 3.1350384
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1911955, 2.1844127
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6478171, 2.6445651
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5405827, 2.5570412
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6729760, 1.6732762
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8554440, 1.8556905
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5608692, 3.5571551
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9746528, 1.9721479
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3731570, 2.3652532

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152856, upper bound: 0.8094257
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8060029, upper bound: 0.8187301
time: 5.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8716888, 1.8728614
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1392279, 3.1342340
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1821070, 2.1934979
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6418586, 2.6505246
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5506916, 2.5469246
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6730990, 1.6731517
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8601122, 1.8510158
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5572720, 3.5607500
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9753413, 1.9714513
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3631148, 2.3752947

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8143183, upper bound: 0.8186933
time: 12.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8152919, upper bound: 0.8177413
time: 6.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8683228, 1.8733010
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1345634, 3.1342869
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1896353, 2.1972046
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6473942, 2.6555052
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5704231, 2.5493469
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6701603, 1.6705236
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8590245, 1.8603764
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5657187, 3.5587478
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9749289, 1.9807439
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3754587, 2.3845844

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4568

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8129582, upper bound: 0.8187346
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8139320, upper bound: 0.8177773
time: 8.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8452830, 1.8569226
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1092148, 3.1166344
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1797786, 2.1864371
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6630373, 2.6680019
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6347151, 2.6266441
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6696754, 1.6675148
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8352313, 1.8428514
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5346231, 3.5246720
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9732962, 1.9731469
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3964381, 2.4022942

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8046420, upper bound: 0.8186845
time: 7.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8046791, upper bound: 0.8186852
time: 15.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8687963, 1.8634796
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1067562, 3.1086478
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1948972, 2.1912041
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6355104, 2.6330178
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6152349, 2.6214900
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6692033, 1.6695039
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8328671, 1.8295834
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5140934, 3.5217571
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9735918, 1.9722915
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.4013143, 2.3943596

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 820

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8173109, upper bound: 0.8061753
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8124710, upper bound: 0.8110168
time: 6.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 26.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8152856, upper bound: 0.8094257
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8060029, upper bound: 0.8187301
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8143183, upper bound: 0.8186933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8152919, upper bound: 0.8177413
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8129582, upper bound: 0.8187346
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8139320, upper bound: 0.8177773
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8046420, upper bound: 0.8186845
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8046791, upper bound: 0.8186852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8173109, upper bound: 0.8061753
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 26.91
Output dim: 0, lower bound: -0.8124710, upper bound: 0.8110168

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8530860, 1.8614683
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1175508, 3.1236138
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1869941, 2.1821074
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6472678, 2.6435633
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5399857, 2.5559440
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6703544, 1.6684866
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8353477, 1.8446882
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5530481, 3.5429010
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9724870, 1.9681921
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3722706, 2.3647661

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 920

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4631

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8060012, upper bound: 0.8181859
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8054608, upper bound: 0.8187283
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8670630, 1.8692546
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1312075, 3.1290259
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1821251, 2.1935427
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6369314, 2.6477876
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5471735, 2.5406122
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6694369, 1.6701012
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8402643, 1.8344712
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5435896, 3.5443234
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9745378, 1.9707808
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3625817, 2.3749976

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8113251, upper bound: 0.8185361
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8141603, upper bound: 0.8157061
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8636956, 1.8696923
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1265421, 3.1290779
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1896544, 2.1972477
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6424646, 2.6527665
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5669022, 2.5430346
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6664977, 1.6674728
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8391790, 1.8438339
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5520334, 3.5423193
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9741230, 1.9800725
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3749261, 2.3842878

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 467
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 5734
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4631

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 467

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8129570, upper bound: 0.8187347
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8129569, upper bound: 0.8186916
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8449907, 1.8558555
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1081095, 3.1163344
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1764159, 2.1739883
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6608295, 2.6598356
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6208515, 2.6228881
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6695046, 1.6674669
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8288188, 1.8411136
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5332861, 3.5197392
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9723406, 1.9728885
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3927331, 2.3885462

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8046420, upper bound: 0.8186846
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8046222, upper bound: 0.8186846
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8442159, 1.8566301
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1089134, 3.1155295
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1673293, 2.1830735
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6548700, 2.6657951
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.6309595, 2.6127791
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6696272, 1.6673439
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8334923, 1.8364389
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5296898, 3.5233359
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9730377, 1.9721918
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3826904, 2.3985877

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4631
type: RSZ, layer: 1, pos: 4568
type: RSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8046791, upper bound: 0.8186874
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8046605, upper bound: 0.8186866
time: 5.45 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 25.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8060012, upper bound: 0.8181859
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8054608, upper bound: 0.8187283
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8113251, upper bound: 0.8185361
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8141603, upper bound: 0.8157061
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8129570, upper bound: 0.8187347
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8129569, upper bound: 0.8186916
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8046420, upper bound: 0.8186846
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8046222, upper bound: 0.8186846
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8046791, upper bound: 0.8186874
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.71
Output dim: 0, lower bound: -0.8046605, upper bound: 0.8186866

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8579459, 1.8649139
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1014519, 3.1101918
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1931067, 2.1893134
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6165142, 2.6179223
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5337915, 2.5456433
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6703041, 1.6693563
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8309031, 1.8387883
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5446424, 3.5358982
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9696450, 1.9658227
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3711014, 2.3623335

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 4568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8029552, upper bound: 0.8180271
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8058293, upper bound: 0.8152257
time: 5.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 8.8957224, 11.2440281, 8.8957224, 11.2440281, -1.8565316, 1.8663282
1: -19.2545967, -15.4455948, -19.2545967, -15.4455948, -3.1041298, 3.1075149
2: -3.8407464, -1.0405811, -3.8407464, -1.0405811, -2.1942005, 2.1882200
3: -13.5667639, -10.0373669, -13.5667639, -10.0373669, -2.6216278, 2.6128087
4: -15.5752630, -11.9897022, -15.5752630, -11.9897022, -2.5296850, 2.5497499
5: -6.1198454, -3.7446783, -6.1198454, -3.7446783, -1.6712239, 1.6684363
6: -3.6156614, -1.3861105, -3.6156614, -1.3861105, -1.8294477, 1.8402438
7: -7.7258506, -3.8985784, -7.7258506, -3.8985784, -3.5460463, 3.5344949
8: -2.8077574, -0.6163578, -2.8077574, -0.6163578, -1.9701180, 1.9653502
9: -9.4011059, -6.0042830, -9.4011059, -6.0042830, -2.3698378, 2.3635972

Time for backsubstitution: 14.40 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.8807220458984375
rel_dist={0: [-0.8193859179682992, 0.8193858024795073]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1662.89 seconds
