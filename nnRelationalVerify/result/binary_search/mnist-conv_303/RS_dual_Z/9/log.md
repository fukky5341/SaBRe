## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.15950791595
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.7245874, 2.7245874)
1: (-7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764)
2: (-6.1131477, -4.0248523, -6.1131477, -4.0248523, -2.0882955, 2.0882955)
3: (-6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959)
4: (-6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728)
5: (-6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860)
6: (-11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779)
7: (2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937)
8: (-4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.3597069, 2.3597069)
9: (-2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474)

## BASE Result
execution time: IAR + LP analysis = 13.81 + 32.99 = 46.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.6885415, upper bound: 1.6885395


# Binary Search by BASE starts (time budget: 3553.20 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.0716936588287354
rel_dist={7: [-1.3781560099392798, 1.3781554825854307]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.974029779434204
rel_dist={7: [-1.163764129436149, 1.1637638135586528]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.9068553447723389
rel_dist={7: [-0.9954076049845186, 0.9954053351782006]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.9404423236846924
rel_dist={7: [-1.086373347751306, 1.0863705612652526]}

## Binary Search Result
Binary search time: 195.51 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3357.69 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4322776, upper bound: 1.4397754
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397756, upper bound: 1.4322772
time: 4.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.47
Output dim: 7, lower bound: -1.4322776, upper bound: 1.4397754
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.47
Output dim: 7, lower bound: -1.4397756, upper bound: 1.4322772

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6295338, 2.6166923
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9117665, 1.9176548
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2592912, 2.2650011
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4322675, upper bound: 1.4304458
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4229362, upper bound: 1.4397655
time: 3.80 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6166925, 2.6295335
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9176545, 1.9117663
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2650013, 2.2592912
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397655, upper bound: 1.4229358
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304461, upper bound: 1.4322677
time: 4.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.10
Output dim: 7, lower bound: -1.4322675, upper bound: 1.4304458
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.10
Output dim: 7, lower bound: -1.4229362, upper bound: 1.4397655
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.10
Output dim: 7, lower bound: -1.4397655, upper bound: 1.4229358
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.10
Output dim: 7, lower bound: -1.4304461, upper bound: 1.4322677

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6289649, 2.6152904
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9047527, 1.9148130
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2578034, 2.2613320
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286767, upper bound: 1.4304415
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4322632, upper bound: 1.4269240
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6281319, 2.6161237
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9089246, 1.9106414
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2556219, 2.2635136
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4193809, upper bound: 1.4397603
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4229320, upper bound: 1.4361864
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6161242, 2.6281316
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9106412, 1.9089246
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2635136, 2.2556221
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4361860, upper bound: 1.4229315
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397600, upper bound: 1.4193805
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6152906, 2.6289649
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9148130, 1.9047530
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2613320, 2.2578037
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4269243, upper bound: 1.4322633
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304419, upper bound: 1.4286770
time: 3.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4286767, upper bound: 1.4304415
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4322632, upper bound: 1.4269240
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4193809, upper bound: 1.4397603
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4229320, upper bound: 1.4361864
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4361860, upper bound: 1.4229315
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4397600, upper bound: 1.4193805
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4269243, upper bound: 1.4322633
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4304419, upper bound: 1.4286770

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6168785, 2.6067238
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9155755, 1.9228928
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2415490, 2.2498090
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286739, upper bound: 1.4282029
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4263096, upper bound: 1.4304391
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6203985, 2.6032040
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9128327, 1.9256358
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2462807, 2.2450774
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4322602, upper bound: 1.4246382
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4298737, upper bound: 1.4269218
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6160450, 2.6075573
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9197474, 1.9187212
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2393675, 2.2519908
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4193781, upper bound: 1.4375793
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4168846, upper bound: 1.4397573
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6195650, 2.6040373
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9170041, 1.9214640
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2440991, 2.2472589
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4229291, upper bound: 1.4340004
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4204470, upper bound: 1.4361834
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6040373, 2.6195650
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9214640, 1.9170043
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2472587, 2.2440991
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4361831, upper bound: 1.4204465
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4340002, upper bound: 1.4229292
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6075573, 2.6160452
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9187212, 1.9197474
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2519908, 2.2393675
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397571, upper bound: 1.4168841
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4375793, upper bound: 1.4193782
time: 4.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6032038, 2.6203985
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9256358, 1.9128325
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2450771, 2.2462807
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4269215, upper bound: 1.4298738
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4246385, upper bound: 1.4322604
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6067238, 2.6168785
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9228926, 1.9155755
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2498093, 2.2415490
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304390, upper bound: 1.4263098
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4282033, upper bound: 1.4286738
time: 4.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4286739, upper bound: 1.4282029
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4263096, upper bound: 1.4304391
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4322602, upper bound: 1.4246382
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4298737, upper bound: 1.4269218
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4193781, upper bound: 1.4375793
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4168846, upper bound: 1.4397573
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4229291, upper bound: 1.4340004
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4204470, upper bound: 1.4361834
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4361831, upper bound: 1.4204465
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4340002, upper bound: 1.4229292
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4397571, upper bound: 1.4168841
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4375793, upper bound: 1.4193782
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4269215, upper bound: 1.4298738
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4246385, upper bound: 1.4322604
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4304390, upper bound: 1.4263098
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.16
Output dim: 7, lower bound: -1.4282033, upper bound: 1.4286738

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6195650, 2.6131902
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9038680, 1.9063635
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2290530, 2.2321708
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4214100, upper bound: 1.4281940
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286649, upper bound: 1.4201801
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6233444, 2.6094103
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8990462, 1.9111845
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2239108, 2.2373126
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4182312, upper bound: 1.4304298
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4263013, upper bound: 1.4232553
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6230845, 2.6096697
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9011242, 1.9091065
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2337842, 2.2274392
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4249638, upper bound: 1.4246293
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4322513, upper bound: 1.4166311
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6268649, 2.6058903
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8963029, 1.9139283
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2286420, 2.2325826
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4217776, upper bound: 1.4269122
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4298653, upper bound: 1.4197143
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6187315, 2.6140234
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9080398, 1.9021916
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2268720, 2.2343526
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4119277, upper bound: 1.4375710
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4193685, upper bound: 1.4295721
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6225109, 2.6102436
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9032180, 1.9070129
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2217288, 2.2394941
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4088090, upper bound: 1.4397484
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4168753, upper bound: 1.4327557
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6222515, 2.6105032
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9052961, 1.9049346
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2316022, 2.2296207
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4154756, upper bound: 1.4339920
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4229195, upper bound: 1.4260113
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6260314, 2.6067238
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9004748, 1.9097564
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2264609, 2.2347643
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4123580, upper bound: 1.4361747
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4204376, upper bound: 1.4291985
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6067238, 2.6260314
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9097564, 1.9004750
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2347646, 2.2264609
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4291984, upper bound: 1.4204379
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4361742, upper bound: 1.4123580
time: 4.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6105032, 2.6222513
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9049346, 1.9052961
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2296205, 2.2316024
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4260112, upper bound: 1.4229199
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4339918, upper bound: 1.4154758
time: 4.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6102433, 2.6225109
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9070127, 1.9032178
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2394938, 2.2217293
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4327561, upper bound: 1.4168756
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4397481, upper bound: 1.4088091
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6140237, 2.6187315
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9021914, 1.9080396
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2343526, 2.2268717
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4295717, upper bound: 1.4193687
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4375709, upper bound: 1.4119277
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6058903, 2.6268647
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9139283, 1.8963032
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2325826, 2.2286425
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4197142, upper bound: 1.4298658
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4269119, upper bound: 1.4217778
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6096697, 2.6230848
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9091065, 1.9011242
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2274394, 2.2337840
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4166312, upper bound: 1.4322515
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4246291, upper bound: 1.4249640
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6094103, 2.6233444
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9111845, 1.8990462
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2373128, 2.2239108
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4232553, upper bound: 1.4263018
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4304294, upper bound: 1.4182315
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6131902, 2.6195650
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9063632, 1.9038680
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2321706, 2.2290533
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4201802, upper bound: 1.4286653
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4281939, upper bound: 1.4214100
time: 4.85 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4214100, upper bound: 1.4281940
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4286649, upper bound: 1.4201801
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4182312, upper bound: 1.4304298
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4263013, upper bound: 1.4232553
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4249638, upper bound: 1.4246293
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4322513, upper bound: 1.4166311
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4217776, upper bound: 1.4269122
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4298653, upper bound: 1.4197143
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4119277, upper bound: 1.4375710
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4193685, upper bound: 1.4295721
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4088090, upper bound: 1.4397484
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4168753, upper bound: 1.4327557
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4154756, upper bound: 1.4339920
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4229195, upper bound: 1.4260113
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4123580, upper bound: 1.4361747
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4204376, upper bound: 1.4291985
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4291984, upper bound: 1.4204379
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4361742, upper bound: 1.4123580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4260112, upper bound: 1.4229199
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4339918, upper bound: 1.4154758
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4327561, upper bound: 1.4168756
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4397481, upper bound: 1.4088091
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4295717, upper bound: 1.4193687
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4375709, upper bound: 1.4119277
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4197142, upper bound: 1.4298658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4269119, upper bound: 1.4217778
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4166312, upper bound: 1.4322515
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4246291, upper bound: 1.4249640
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4232553, upper bound: 1.4263018
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4304294, upper bound: 1.4182315
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4201802, upper bound: 1.4286653
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.85
Output dim: 7, lower bound: -1.4281939, upper bound: 1.4214100

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5949950, 2.6047232
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8904588, 1.9017491
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2201266, 2.2062814
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4214094, upper bound: 1.4276690
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4208388, upper bound: 1.4281936
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6110978, 2.5886204
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8992541, 1.8929543
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2031636, 2.2232442
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4286649, upper bound: 1.4195713
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4282146, upper bound: 1.4201801
time: 4.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5987749, 2.6009433
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8856370, 1.9065702
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2149844, 2.2114229
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.10 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.0716936588287354
rel_dist={7: [-1.4397835588004604, 1.43978321464756]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353226, upper bound: 1.2399058
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2399061, upper bound: 1.2353222
time: 6.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.72
Output dim: 7, lower bound: -1.2353226, upper bound: 1.2399058
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.72
Output dim: 7, lower bound: -1.2399061, upper bound: 1.2353222

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4203453, 2.4130077
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7824836, 1.7858484
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4678621, 2.4675519
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1138775, 2.1079307
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7148328, 2.7170906
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9925082, 1.9981358
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0822506, 2.0855136
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353159, upper bound: 1.2341981
time: 7.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2296119, upper bound: 1.2398992
time: 4.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4130077, 2.4203453
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7858481, 1.7824836
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4675522, 2.4678621
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1079309, 2.1138778
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7170911, 2.7148333
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9981358, 1.9925084
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0855136, 2.0822506
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398995, upper bound: 1.2296115
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341987, upper bound: 1.2353156
time: 8.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 7, lower bound: -1.2353159, upper bound: 1.2341981
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 7, lower bound: -1.2296119, upper bound: 1.2398992
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 7, lower bound: -1.2398995, upper bound: 1.2296115
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 7, lower bound: -1.2341987, upper bound: 1.2353156

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4194198, 2.4116058
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7754703, 1.7812190
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4662838, 2.4651606
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1075990, 2.1037853
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7131000, 2.7144651
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9871523, 1.9900219
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0798283, 2.0818443
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2339814, upper bound: 1.2341970
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353143, upper bound: 1.2328577
time: 7.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4189434, 2.4120817
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7778540, 1.7788351
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4654708, 2.4659734
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1097324, 2.1016519
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7122083, 2.7153573
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9843948, 1.9927797
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0785818, 2.0830908
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2282669, upper bound: 1.2398979
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2296102, upper bound: 1.2385645
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4120817, 2.4189434
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7788353, 1.7778542
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4659734, 2.4654708
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1016519, 2.1097322
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7153573, 2.7122078
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9927795, 1.9843946
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0830913, 2.0785818
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2385639, upper bound: 1.2296095
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398979, upper bound: 1.2282666
time: 7.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4116058, 2.4194198
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7812190, 1.7754703
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4651608, 2.4662838
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1037853, 2.1075985
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7144647, 2.7131000
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9900219, 1.9871523
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0818443, 2.0798283
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2328584, upper bound: 1.2353141
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341971, upper bound: 1.2339813
time: 7.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2339814, upper bound: 1.2341970
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2353143, upper bound: 1.2328577
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2282669, upper bound: 1.2398979
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2296102, upper bound: 1.2385645
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2385639, upper bound: 1.2296095
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2398979, upper bound: 1.2282666
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2328584, upper bound: 1.2353141
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.04
Output dim: 7, lower bound: -1.2341971, upper bound: 1.2339813

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4073334, 2.4015307
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7851171, 1.7892988
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4678106, 2.4664485
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1104026, 2.1103368
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7077122, 2.7079988
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9859982, 1.9890604
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0635738, 2.0682936
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2339748, upper bound: 1.2299375
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2297201, upper bound: 1.2341903
time: 4.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4093447, 2.3995194
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7835498, 1.7908661
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4675717, 2.4666877
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1141505, 2.1065893
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7066336, 2.7090769
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9861908, 1.9888680
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0662775, 2.0655897
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353075, upper bound: 1.2286120
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2310470, upper bound: 1.2328514
time: 7.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4068570, 2.4020066
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2429864
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7875013, 1.7869148
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4669976, 2.4672613
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1125360, 2.1082032
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7068195, 2.7088909
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9832406, 1.9918182
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0623269, 2.0695403
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2282602, upper bound: 1.2356647
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2239939, upper bound: 1.2398934
time: 6.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4088683, 2.3999953
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7859340, 1.7884822
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4667587, 2.4675004
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1162839, 2.1044559
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7057419, 2.7099690
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9834332, 1.9916258
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0650311, 2.0668364
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2296032, upper bound: 1.2343374
time: 8.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2253194, upper bound: 1.2385573
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3999953, 2.4088683
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7884822, 1.7859337
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4675002, 2.4667587
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1044559, 2.1162839
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7099686, 2.7057414
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9916258, 1.9834332
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0668364, 2.0650308
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2385573, upper bound: 1.2253189
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2343376, upper bound: 1.2296032
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4020071, 2.4068570
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2429867, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7869148, 1.7875013
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4672613, 2.4669979
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1082034, 2.1125362
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7088909, 2.7068200
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9918180, 1.9832406
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0695405, 2.0623269
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398911, upper bound: 1.2239931
time: 5.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2356648, upper bound: 1.2282598
time: 6.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3995190, 2.4093447
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7908659, 1.7835500
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4666877, 2.4675717
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1065893, 2.1141503
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7090769, 2.7066336
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9888678, 1.9861910
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0655899, 2.0662775
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2328518, upper bound: 1.2310466
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2286126, upper bound: 1.2353075
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4015307, 2.4073334
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7892985, 1.7851174
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4664483, 2.4678106
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1103368, 2.1104026
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7079983, 2.7077122
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9890604, 1.9859984
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0682936, 2.0635736
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341903, upper bound: 1.2297194
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2299375, upper bound: 1.2339743
time: 6.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2339748, upper bound: 1.2299375
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2297201, upper bound: 1.2341903
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2353075, upper bound: 1.2286120
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2310470, upper bound: 1.2328514
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2282602, upper bound: 1.2356647
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2239939, upper bound: 1.2398934
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2296032, upper bound: 1.2343374
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2253194, upper bound: 1.2385573
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2385573, upper bound: 1.2253189
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2343376, upper bound: 1.2296032
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2398911, upper bound: 1.2239931
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2356648, upper bound: 1.2282598
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2328518, upper bound: 1.2310466
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2286126, upper bound: 1.2353075
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2341903, upper bound: 1.2297194
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.16
Output dim: 7, lower bound: -1.2299375, upper bound: 1.2339743

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4100194, 2.4063768
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2115951, 2.2142479
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7713430, 1.7727692
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4653702, 2.4600687
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0997019, 2.1014166
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6935043, 2.6996613
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9847765, 1.9858510
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0488734, 2.0506556
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2290169, upper bound: 1.2299317
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2339698, upper bound: 1.2250056
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4121795, 2.4042168
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2177405, 2.2081025
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7685878, 1.7755241
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4614310, 2.4640079
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1014824, 2.0996356
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6993752, 2.6937919
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9827890, 1.9878385
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0459352, 2.0535934
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2247881, upper bound: 1.2341845
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2297151, upper bound: 1.2292534
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4120307, 2.4043651
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2081318, 2.2177110
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7697752, 1.7743366
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4651313, 2.4603076
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1034493, 2.0976691
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6924267, 2.7007394
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9849691, 1.9856584
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0515771, 2.0479517
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2303705, upper bound: 1.2286065
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353025, upper bound: 1.2236797
time: 4.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4141908, 2.4022055
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2142773, 2.2115655
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7670205, 1.7770920
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4611917, 2.4642472
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1052303, 2.0958884
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6982956, 2.6948700
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9829817, 1.9876459
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0486388, 2.0508907
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2261157, upper bound: 1.2328462
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2310419, upper bound: 1.2278943
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4095435, 2.4068532
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2136450, 2.2121983
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7737272, 1.7703853
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4645576, 2.4608812
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1018353, 2.0992832
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6926126, 2.7005534
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9820185, 1.9886086
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0476270, 2.0519021
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2233091, upper bound: 1.2356593
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2282546, upper bound: 1.2307333
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4117031, 2.4046931
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2197905, 2.2060528
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7709715, 1.7731404
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4606180, 2.4648209
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1036158, 2.0975022
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6984825, 2.6946840
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9800310, 1.9905961
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0446887, 2.0548401
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2190620, upper bound: 1.2398858
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2239883, upper bound: 1.2349538
time: 5.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4115548, 2.4048414
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2101817, 2.2156613
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7721593, 1.7719529
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4643183, 2.4611206
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1055827, 2.0955355
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6915340, 2.7016320
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9822111, 1.9884160
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0503306, 2.0491982
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2246682, upper bound: 1.2343321
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2295976, upper bound: 1.2294043
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4137149, 2.4026818
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2163272, 2.2095158
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7694042, 1.7747080
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4603786, 2.4650600
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1073637, 2.0937548
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6974039, 2.6957622
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9802237, 1.9904034
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0473924, 2.0521374
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2203877, upper bound: 1.2385522
time: 5.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2253139, upper bound: 1.2336022
time: 6.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4026818, 2.4137149
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2095160, 2.2163272
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7747080, 1.7694044
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4650602, 2.4603786
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0937548, 2.1073637
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6957626, 2.6974044
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9904032, 1.9802239
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0521369, 2.0473926
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2336021, upper bound: 1.2253138
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2385523, upper bound: 1.2203873
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4048414, 2.4115548
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2156610, 2.2101817
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7719529, 1.7721593
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4611206, 2.4643183
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0955353, 2.1055827
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7016315, 2.6915345
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9884157, 1.9822114
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0491986, 2.0503306
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2294045, upper bound: 1.2295999
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2343326, upper bound: 1.2246678
time: 8.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4046931, 2.4117031
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2060528, 2.2197902
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7731402, 1.7709718
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4648209, 2.4606180
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0975022, 2.1036160
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6946831, 2.6984825
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9905958, 1.9800313
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0548396, 2.0446889
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349540, upper bound: 1.2239880
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2398861, upper bound: 1.2190617
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4068532, 2.4095435
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2121983, 2.2136447
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7703855, 1.7737269
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4608812, 2.4645574
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0992832, 2.1018353
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7005539, 2.6926131
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9886084, 1.9820187
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0519023, 2.0476272
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2307335, upper bound: 1.2282542
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2356598, upper bound: 1.2233090
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4022055, 2.4141908
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2115655, 2.2142775
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7770917, 1.7670205
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4642472, 2.4611917
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0958881, 2.1052303
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6948700, 2.6982961
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9876461, 1.9829814
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0508904, 2.0486393
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2278943, upper bound: 1.2310420
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2328462, upper bound: 1.2261160
time: 4.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4043651, 2.4120307
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2177110, 2.2081320
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7743366, 1.7697754
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4603081, 2.4651310
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0976691, 2.1034493
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7007389, 2.6924267
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9856586, 1.9849689
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0479512, 2.0515773
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2236806, upper bound: 1.2353025
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2286071, upper bound: 1.2303704
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4042168, 2.4121795
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2081022, 2.2177405
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7755239, 1.7685878
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4640083, 2.4614308
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0996356, 2.1014824
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6937923, 2.6993747
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9878387, 1.9827888
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0535932, 2.0459354
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2292533, upper bound: 1.2297148
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2341848, upper bound: 1.2247877
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4063768, 2.4100194
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2142482, 2.2115951
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7727692, 1.7713432
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4600687, 2.4653702
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1014166, 2.0997016
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6996613, 2.6935048
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9858513, 1.9847763
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0506558, 2.0488739
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.0076169967651367
rel_dist={7: [-1.2399111761101596, 1.2399105900004943]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602918, upper bound: 1.1637594
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602940, upper bound: 1.1602913
time: 4.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.07
Output dim: 7, lower bound: -1.1602918, upper bound: 1.1637594
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.07
Output dim: 7, lower bound: -1.1602940, upper bound: 1.1602913

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3506160, 2.3451128
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2213068, 2.2197475
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7393889, 1.7419131
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4092641, 2.4090312
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0519125, 2.0474524
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6616406, 2.6633334
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9589212, 1.9631417
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0232372, 2.0256844
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602864, upper bound: 1.1595187
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1560506, upper bound: 1.1637545
time: 5.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3451128, 2.3506160
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2197475, 2.2213068
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7419133, 1.7393894
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4090314, 2.4092638
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0474527, 2.0519128
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6633334, 2.6616402
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9631417, 1.9589212
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0256844, 2.0232372
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637548, upper bound: 1.1560506
time: 8.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1595190, upper bound: 1.1602862
time: 5.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 28.79 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.79
Output dim: 7, lower bound: -1.1602864, upper bound: 1.1595187
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.79
Output dim: 7, lower bound: -1.1560506, upper bound: 1.1637545
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.79
Output dim: 7, lower bound: -1.1637548, upper bound: 1.1560506
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.79
Output dim: 7, lower bound: -1.1595190, upper bound: 1.1602862

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3495712, 2.3437109
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2152729, 2.2152507
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7323761, 1.7366877
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4074821, 2.4066398
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0456340, 2.0427735
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6596837, 2.6607080
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9528759, 1.9550278
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0205030, 2.0220151
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1593227, upper bound: 1.1595175
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602852, upper bound: 1.1585555
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3492141, 2.3440681
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2168102, 2.2137134
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7341638, 1.7348998
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4068727, 2.4072495
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0472343, 2.0411735
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6590142, 2.6613770
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9508073, 1.9570961
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0195680, 2.0229502
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1550870, upper bound: 1.1637533
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1560518, upper bound: 1.1627911
time: 9.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3440681, 2.3492141
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2137136, 2.2168102
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7348995, 1.7341640
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4072495, 2.4068725
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0411737, 2.0472338
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6613774, 2.6590147
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9570963, 1.9508073
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0229502, 2.0195680
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1627912, upper bound: 1.1560491
time: 9.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637536, upper bound: 1.1550868
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3437109, 2.3495712
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2152510, 2.2152729
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7366877, 1.7323761
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4066401, 2.4074821
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0427740, 2.0456336
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6607080, 2.6596842
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9550278, 1.9528756
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0220151, 2.0205030
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1585554, upper bound: 1.1602849
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1595178, upper bound: 1.1593226
time: 5.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1593227, upper bound: 1.1595175
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1602852, upper bound: 1.1585555
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1550870, upper bound: 1.1637533
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1560518, upper bound: 1.1627911
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1627912, upper bound: 1.1560491
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1637536, upper bound: 1.1550868
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1585554, upper bound: 1.1602849
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.02
Output dim: 7, lower bound: -1.1595178, upper bound: 1.1593226

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3374848, 2.3331327
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1971126, 2.1944931
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7416310, 1.7447672
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4089494, 2.4079278
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0484376, 2.0483882
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6540260, 2.6542416
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9517217, 1.9540184
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0042486, 2.0077887
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1593173, upper bound: 1.1561004
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1558851, upper bound: 1.1595121
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3389935, 2.3316245
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1945152, 2.1970904
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7404556, 1.7459428
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4087701, 2.4081070
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0512486, 2.0455775
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6532173, 2.6550503
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9518661, 1.9538739
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0062761, 2.0057607
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602796, upper bound: 1.1551232
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1568620, upper bound: 1.1585494
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3371277, 2.3334899
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1986499, 2.1929557
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7434192, 1.7429793
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4083395, 2.4085374
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0500379, 2.0467882
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6533575, 2.6549106
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9496531, 1.9560866
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0033135, 2.0087233
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1550816, upper bound: 1.1603344
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1516510, upper bound: 1.1637487
time: 4.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3386364, 2.3319817
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1960526, 2.1955531
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7422438, 1.7441549
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4081602, 2.4087167
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0528483, 2.0439775
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6525478, 2.6557193
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9497976, 1.9559422
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0053415, 2.0066953
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1560438, upper bound: 1.1593572
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1526279, upper bound: 1.1627857
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3319817, 2.3386364
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1955528, 2.1960526
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7441549, 1.7422438
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4087167, 2.4081604
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0439773, 2.0528486
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6557198, 2.6525488
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9559422, 1.9497979
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0066957, 2.0053415
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1627858, upper bound: 1.1526277
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1593581, upper bound: 1.1560433
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3334899, 2.3371277
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1929555, 2.1986499
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7429795, 1.7434192
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4085374, 2.4083397
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0467882, 2.0500379
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6549110, 2.6533575
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9560862, 1.9496534
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0087233, 2.0033135
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637480, upper bound: 1.1516501
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1603348, upper bound: 1.1550809
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3316245, 2.3389935
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1970901, 2.1945152
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7459426, 1.7404556
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4081068, 2.4087701
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0455775, 2.0512486
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6550503, 2.6532178
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9538736, 1.9518661
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0057607, 2.0062761
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1585500, upper bound: 1.1568617
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1551240, upper bound: 1.1602794
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3331327, 2.3374848
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1944928, 2.1971126
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7447672, 1.7416313
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4079275, 2.4089494
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0483880, 2.0484376
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6542416, 2.6540265
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9540181, 1.9517217
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0077887, 2.0042486
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1595122, upper bound: 1.1558844
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1561005, upper bound: 1.1593167
time: 4.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1593173, upper bound: 1.1561004
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1558851, upper bound: 1.1595121
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1602796, upper bound: 1.1551232
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1568620, upper bound: 1.1585494
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1550816, upper bound: 1.1603344
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1516510, upper bound: 1.1637487
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1560438, upper bound: 1.1593572
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1526279, upper bound: 1.1627857
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1627858, upper bound: 1.1526277
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1593581, upper bound: 1.1560433
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1637480, upper bound: 1.1516501
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1603348, upper bound: 1.1550809
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1585500, upper bound: 1.1568617
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1551240, upper bound: 1.1602794
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1595122, upper bound: 1.1558844
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.59
Output dim: 7, lower bound: -1.1561005, upper bound: 1.1593167

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3417912, 2.3358192
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1647878, 2.1575592
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7251017, 1.7303040
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4025698, 2.4045024
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4243555, 2.4151063
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0390720, 2.0376871
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6442223, 2.6400342
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9485121, 1.9522994
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9866099, 1.9923537
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1521849, upper bound: 1.1595078
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1558813, upper bound: 1.1558133
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3416796, 2.3359303
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1575813, 2.1647656
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7259924, 1.7294135
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4053450, 2.4017272
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4178047, 2.4216566
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0405474, 2.0362122
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6390104, 2.6452451
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9501476, 1.9506643
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9908414, 1.9881222
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1565800, upper bound: 1.1551196
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602759, upper bound: 1.1514239
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3398137, 2.3377962
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1617160, 2.1606309
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7289560, 1.7264500
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4049144, 2.4021573
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4204855, 2.4189758
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0393367, 2.0374227
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6391506, 2.6451058
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9479342, 1.9528770
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9878793, 1.9910853
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1513822, upper bound: 1.1603308
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1550775, upper bound: 1.1566342
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3414340, 2.3361764
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1663251, 2.1560218
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7268898, 1.7285161
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4019599, 2.4051118
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4261999, 2.4132609
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0406723, 2.0360870
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6435528, 2.6407037
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9464436, 1.9543674
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9856753, 1.9932888
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1479509, upper bound: 1.1637438
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1516469, upper bound: 1.1600475
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3429422, 2.3346677
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1637278, 2.1586192
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7257140, 1.7296920
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4017806, 2.4052913
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4253645, 2.4140968
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0434833, 2.0332766
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6427441, 2.6415124
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9465885, 1.9542232
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9877028, 1.9912617
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1489280, upper bound: 1.1627816
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1526238, upper bound: 1.1590783
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3346677, 2.3429422
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1586194, 2.1637278
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7296917, 1.7257142
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4052916, 2.4017804
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4140968, 2.4253645
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0332766, 2.0434833
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6415119, 2.6427436
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9542236, 1.9465883
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9912620, 1.9877031
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1590788, upper bound: 1.1526245
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1627820, upper bound: 1.1489278
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3361764, 2.3414340
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1560221, 2.1663251
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7285163, 1.7268898
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4051123, 2.4019599
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4132605, 2.4262004
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0360870, 2.0406723
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6407032, 2.6435523
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9543676, 1.9464438
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9932885, 1.9856756
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1600480, upper bound: 1.1516478
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637443, upper bound: 1.1479505
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3377962, 2.3398137
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1606312, 2.1617160
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7264497, 1.7289562
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4021578, 2.4049144
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4189758, 2.4204855
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0374227, 2.0393367
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6451054, 2.6391501
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9528770, 1.9479344
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9910855, 1.9878790
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1566348, upper bound: 1.1550778
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1603310, upper bound: 1.1513816
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3359303, 2.3416796
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1647658, 2.1575813
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7294133, 1.7259924
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4017272, 2.4053445
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4216566, 2.4178047
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0362120, 2.0405474
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6452446, 2.6390109
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9506645, 1.9501472
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9881225, 1.9908416
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1479531, upper bound: 1.1602754
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1551199, upper bound: 1.1565799
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3358192, 2.3417912
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1575594, 2.1647878
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7303040, 1.7251019
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4045024, 2.4025693
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4151068, 2.4243550
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0376873, 2.0390723
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6400337, 2.6442218
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9522991, 1.9485121
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9923539, 1.9866102
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1558136, upper bound: 1.1558808
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1595082, upper bound: 1.1521843
time: 4.63 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1521849, upper bound: 1.1595078
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1558813, upper bound: 1.1558133
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1565800, upper bound: 1.1551196
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1602759, upper bound: 1.1514239
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1513822, upper bound: 1.1603308
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1550775, upper bound: 1.1566342
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1479509, upper bound: 1.1637438
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1516469, upper bound: 1.1600475
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1489280, upper bound: 1.1627816
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1526238, upper bound: 1.1590783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1590788, upper bound: 1.1526245
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1627820, upper bound: 1.1489278
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1600480, upper bound: 1.1516478
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1637443, upper bound: 1.1479505
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1566348, upper bound: 1.1550778
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1603310, upper bound: 1.1513816
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1479531, upper bound: 1.1602754
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1551199, upper bound: 1.1565799
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1558136, upper bound: 1.1558808
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.89
Output dim: 7, lower bound: -1.1595082, upper bound: 1.1521843

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3240113, 2.3113608
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1355281, 2.1340733
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7163527, 1.7160044
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3920512, 2.3832493
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4129610, 2.4149189
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0303526, 2.0288799
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6361275, 2.6412444
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9461985, 1.9451740
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9649520, 1.9695027
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7218251, 1.7230222

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1602756, upper bound: 1.1510984
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1599405, upper bound: 1.1514234
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3152442, 2.3201275
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1310239, 2.1385779
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7155473, 1.7168105
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3864365, 2.3888640
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4137468, 2.4141326
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0320044, 2.0272281
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6351490, 2.6422224
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9424438, 1.9489286
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9692597, 1.9651957
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7282839, 1.7165635

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1513819, upper bound: 1.1600061
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1510467, upper bound: 1.1603308
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3168640, 2.3185077
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1356325, 2.1339688
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7134807, 1.7188766
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3834820, 2.3918185
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4194622, 2.4084177
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0333395, 2.0258923
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6395512, 2.6378202
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9409533, 1.9504192
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9670558, 1.9673994
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7260594, 1.7187879

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1479507, upper bound: 1.1634071
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1476263, upper bound: 1.1637437
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3237653, 2.3116069
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1442719, 2.1253295
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7172496, 1.7151072
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3886666, 2.3866339
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4213572, 2.4065232
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0304775, 2.0287545
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6406689, 2.6367030
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9424949, 1.9488771
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9597859, 1.9746690
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7196050, 1.7252424

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1516467, upper bound: 1.1597103
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1513223, upper bound: 1.1600471
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3183727, 2.3169994
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1330357, 2.1365662
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7123053, 1.7200525
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3833027, 2.3919978
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4186268, 2.4092536
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0361509, 2.0230818
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6387424, 2.6386290
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9410977, 1.9502747
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9690833, 1.9653723
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7258096, 1.7190378

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1489278, upper bound: 1.1624464
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1486035, upper bound: 1.1627816
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3169990, 2.3183727
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1365662, 2.1330354
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7200525, 1.7123053
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3919978, 2.3833025
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4092531, 2.4186268
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0230818, 2.0361507
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6386290, 2.6387429
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9502745, 1.9410980
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9653726, 1.9690833
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7190375, 1.7258098

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1627817, upper bound: 1.1486024
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1624466, upper bound: 1.1489268
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3116069, 2.3237653
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1253295, 2.1442719
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7151072, 1.7172499
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3866339, 2.3886666
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4065237, 2.4213572
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0287542, 2.0304775
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6367035, 2.6406693
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9488773, 1.9424951
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9746690, 1.9597857
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7252421, 1.7196052

Time for backsubstitution: 14.13 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.974029779434204
rel_dist={7: [-1.163764129436149, 1.1637638135586528]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2423.80 seconds
