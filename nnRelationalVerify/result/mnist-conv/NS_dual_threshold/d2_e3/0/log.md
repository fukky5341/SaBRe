## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.8648028314999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.0050335, -4.3815689, -7.0050335, -4.3815689, -1.8557234, 1.8557234)
1: (-9.1259270, -6.9853754, -9.1259270, -6.9853754, -1.7110176, 1.7110167)
2: (-7.6842809, -5.9733133, -7.6842809, -5.9733133, -1.4690595, 1.4690595)
3: (-5.6812768, -3.6317198, -5.6812768, -3.6317198, -1.9553661, 1.9553661)
4: (-9.2629375, -7.2492046, -9.2629375, -7.2492046, -1.6807356, 1.6807356)
5: (1.3427689, 2.7021751, 1.3427689, 2.7021751, -1.1920395, 1.1920395)
6: (-1.6153922, 0.3814738, -1.6153922, 0.3814738, -1.4119353, 1.4119353)
7: (-10.3737221, -8.7644253, -10.3737221, -8.7644253, -1.3383250, 1.3383250)
8: (5.5917873, 7.2588768, 5.5917873, 7.2588768, -1.6670895, 1.6670895)
9: (-5.3475718, -3.8932147, -5.3475718, -3.8932147, -1.2933002, 1.2932997)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.64 + 34.97 = 57.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.8656685, upper bound: 0.8656684

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 466
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 466

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656673, upper bound: 0.8643721
time: 4.99 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656673, upper bound: 0.8656671
time: 4.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.96 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 9.96
Output dim: 8, lower bound: -0.8656673, upper bound: 0.8643721
NS_B2, status: Status.UNKNOWN, split count: 1, time: 9.96
Output dim: 8, lower bound: -0.8656673, upper bound: 0.8656671

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -7.0032077, -4.3922615, -6.9923625, -4.4045224, -1.8313875, 1.8329554
1: -9.1220360, -6.9858894, -9.1165237, -6.9893847, -1.7028837, 1.7014542
2: -7.6817751, -5.9737339, -7.6779971, -5.9761000, -1.4630942, 1.4623289
3: -5.6739497, -3.6329525, -5.6649971, -3.6399214, -1.9401264, 1.9381590
4: -9.2620096, -7.2500124, -9.2608051, -7.2512565, -1.6762133, 1.6762180
5: 1.3442516, 2.6982968, 1.3492622, 2.6939077, -1.1828275, 1.1827688
6: -1.6144762, 0.3792214, -1.6116052, 0.3763576, -1.4060202, 1.4054127
7: -10.3730383, -8.7674570, -10.3699446, -8.7711124, -1.3310165, 1.3314853
8: 5.5955219, 7.2579837, 5.6001821, 7.2536922, -1.6581702, 1.6578016
9: -5.3456683, -3.8934786, -5.3432927, -3.8945012, -1.2892466, 1.2889743

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8617949
time: 4.49 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8643642
time: 4.57 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -7.0050335, -4.3815699, -7.0050306, -4.3815761, -1.8434649, 1.8557205
1: -9.1259260, -6.9853725, -9.1259251, -6.9853754, -1.7110157, 1.7105680
2: -7.6842794, -5.9733133, -7.6842775, -5.9733143, -1.4696121, 1.4690018
3: -5.6812754, -3.6317205, -5.6812730, -3.6317220, -1.9553652, 1.9548931
4: -9.2629375, -7.2492070, -9.2629366, -7.2492065, -1.6782379, 1.6855278
5: 1.3427700, 2.7021759, 1.3427701, 2.7021732, -1.1916509, 1.1920371
6: -1.6153920, 0.3814735, -1.6153905, 0.3814723, -1.4118423, 1.4128175
7: -10.3737221, -8.7644262, -10.3737202, -8.7644281, -1.3348293, 1.3383241
8: 5.5917873, 7.2588763, 5.5917883, 7.2588758, -1.6670885, 1.6670880
9: -5.3475714, -3.8932152, -5.3475714, -3.8932142, -1.2916532, 1.2966990

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8656596
time: 6.17 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656594, upper bound: 0.8656599
time: 4.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.85 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8617949
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 8, lower bound: -0.8656599, upper bound: 0.8643642
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8656596
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 31.85
Output dim: 8, lower bound: -0.8656594, upper bound: 0.8656599

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -7.0006485, -4.4015336, -6.9920444, -4.4051771, -1.8001089, 1.8234024
1: -9.1197348, -6.9867721, -9.1163425, -6.9894714, -1.7002506, 1.7001109
2: -7.6803255, -5.9811134, -7.6778502, -5.9766145, -1.4423790, 1.4537773
3: -5.6731396, -3.6373897, -5.6649237, -3.6403775, -1.9386644, 1.9422693
4: -9.2548599, -7.2511616, -9.2603035, -7.2514000, -1.6679010, 1.6692019
5: 1.3488011, 2.6972213, 1.3495793, 2.6937702, -1.1766057, 1.1609278
6: -1.6122735, 0.3645661, -1.6113365, 0.3753309, -1.3759632, 1.3904748
7: -10.3687239, -8.7682056, -10.3696299, -8.7712460, -1.3249550, 1.3127418
8: 5.5988860, 7.2529755, 5.6004710, 7.2533379, -1.6544518, 1.6525044
9: -5.3381839, -3.8942430, -5.3427658, -3.8946083, -1.2806602, 1.2682209

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8617945
time: 4.14 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8617946
time: 5.60 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -7.0483522, -4.3902607, -6.9923406, -4.4045591, -1.8714275, 1.8346252
1: -9.1268787, -6.9777589, -9.1165123, -6.9893880, -1.7082806, 1.7085967
2: -7.7184315, -5.9692249, -7.6779871, -5.9761310, -1.4783149, 1.4660525
3: -5.6810465, -3.6160231, -5.6649914, -3.6399539, -1.9476280, 1.9546118
4: -9.2677336, -7.2119231, -9.2607632, -7.2512670, -1.6812592, 1.6967554
5: 1.3417952, 2.7147775, 1.3492861, 2.6939001, -1.1838756, 1.2017531
6: -1.6823182, 0.3828030, -1.6115861, 0.3762805, -1.4372663, 1.4087896
7: -10.3841524, -8.7436180, -10.3699293, -8.7711201, -1.3397942, 1.3494713
8: 5.5702791, 7.2623529, 5.6002011, 7.2536721, -1.6833930, 1.6621518
9: -5.3538647, -3.8580251, -5.3432584, -3.8945098, -1.2963629, 1.3022087

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656581, upper bound: 0.8640749
time: 4.56 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656553, upper bound: 0.8643595
time: 4.73 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -7.0047140, -4.3822231, -7.0024652, -4.3908377, -1.8339109, 1.8244152
1: -9.1257448, -6.9854631, -9.1236172, -6.9862590, -1.7096729, 1.7079525
2: -7.6841350, -5.9738283, -7.6828289, -5.9806938, -1.4610610, 1.4482889
3: -5.6812048, -3.6321781, -5.6804600, -3.6361580, -1.9594870, 1.9534311
4: -9.2624369, -7.2493496, -9.2557964, -7.2503567, -1.6712933, 1.6772223
5: 1.3430861, 2.7020373, 1.3473229, 2.7010994, -1.1697130, 1.1858077
6: -1.6151252, 0.3804479, -1.6131909, 0.3668208, -1.3969083, 1.3826861
7: -10.3734064, -8.7645597, -10.3694115, -8.7651777, -1.3160515, 1.3322644
8: 5.5920763, 7.2585211, 5.5951519, 7.2538681, -1.6617918, 1.6633692
9: -5.3470440, -3.8933220, -5.3400879, -3.8939795, -1.2709007, 1.2881160

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8630909
time: 7.14 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8656596
time: 5.58 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -7.0050077, -4.3816047, -7.0502138, -4.3795743, -1.8451338, 1.8874617
1: -9.1259117, -6.9853802, -9.1307697, -6.9772487, -1.7181568, 1.7159619
2: -7.6842704, -5.9733448, -7.7209368, -5.9688120, -1.4733310, 1.4843004
3: -5.6812725, -3.6317532, -5.6883712, -3.6148036, -1.9718142, 1.9623995
4: -9.2628937, -7.2492151, -9.2687073, -7.2111220, -1.6987185, 1.6905727
5: 1.3427939, 2.7021670, 1.3402811, 2.7186656, -1.2114754, 1.1931257
6: -1.6153758, 0.3813972, -1.6832237, 0.3850563, -1.4152260, 1.4425211
7: -10.3737068, -8.7644358, -10.3848076, -8.7405872, -1.3539052, 1.3470788
8: 5.5918069, 7.2588525, 5.5665455, 7.2632251, -1.6714182, 1.6923070
9: -5.3475356, -3.8932207, -5.3557844, -3.8577619, -1.3044243, 1.3038116

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653701, upper bound: 0.8656572
time: 4.57 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656546, upper bound: 0.8656552
time: 4.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.29 seconds
NS_B1_A1_B1, status: Status.VERIFIED, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8617945
NS_B1_A1_B2, status: Status.VERIFIED, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8617946
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8656581, upper bound: 0.8640749
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8656553, upper bound: 0.8643595
NS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8630909
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8630912, upper bound: 0.8656596
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8653701, upper bound: 0.8656572
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 8, lower bound: -0.8656546, upper bound: 0.8656552

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.0483532, -4.3902607, -6.9907331, -4.4046822, -1.8713036, 1.8330269
1: -9.1268797, -6.9777603, -9.1158962, -6.9896421, -1.7079792, 1.7079811
2: -7.7184315, -5.9692264, -7.6772542, -5.9763989, -1.4774227, 1.4644842
3: -5.6810465, -3.6160214, -5.6646557, -3.6427443, -1.9447556, 1.9541998
4: -9.2677336, -7.2119231, -9.2597837, -7.2515478, -1.6805191, 1.6953363
5: 1.3417952, 2.7147779, 1.3495071, 2.6923857, -1.1823006, 1.2015162
6: -1.6823187, 0.3828020, -1.6112554, 0.3761344, -1.4371152, 1.4083695
7: -10.3841515, -8.7436180, -10.3696766, -8.7713022, -1.3388515, 1.3487754
8: 5.5702810, 7.2623529, 5.6009121, 7.2534466, -1.6831656, 1.6614408
9: -5.3538642, -3.8580246, -5.3429222, -3.8946815, -1.2958527, 1.3014817

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656542, upper bound: 0.8626221
time: 4.40 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656533, upper bound: 0.8640711
time: 4.57 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.0483494, -4.3902626, -6.9986901, -4.3827791, -1.8739634, 1.8420048
1: -9.1268749, -6.9777613, -9.1192617, -6.9786434, -1.7194557, 1.7120214
2: -7.7184315, -5.9692268, -7.6826825, -5.9633436, -1.4816546, 1.4825282
3: -5.6810446, -3.6160312, -5.7017040, -3.6327910, -1.9558449, 1.9700923
4: -9.2677307, -7.2119246, -9.2656918, -7.2324133, -1.7016783, 1.7016516
5: 1.3417947, 2.7147763, 1.3275609, 2.6957755, -1.1871872, 1.2060194
6: -1.6823180, 0.3828011, -1.6167257, 0.3802285, -1.4395442, 1.4154382
7: -10.3841505, -8.7436199, -10.3803902, -8.7669315, -1.3540440, 1.3545928
8: 5.5702829, 7.2623506, 5.5930767, 7.2576332, -1.6873503, 1.6692739
9: -5.3538628, -3.8580251, -5.3490353, -3.8871336, -1.3088536, 1.3076706

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8642013, upper bound: 0.8643556
time: 4.91 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656513, upper bound: 0.8643555
time: 4.50 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0501032, -4.3795738, -7.0024652, -4.3908377, -1.8725004, 1.8267298
1: -9.1307659, -6.9772601, -9.1236172, -6.9862590, -1.7149601, 1.7149506
2: -7.7208853, -5.9688129, -7.6828289, -5.9806938, -1.4745579, 1.4533682
3: -5.6883688, -3.6148555, -5.6804600, -3.6361580, -1.9633598, 1.9693117
4: -9.2687044, -7.2111635, -9.2557964, -7.2503567, -1.6778336, 1.6952205
5: 1.3402812, 2.7186031, 1.3473229, 2.7010994, -1.1721702, 1.2008462
6: -1.6830883, 0.3850541, -1.6131909, 0.3668208, -1.4278731, 1.3870420
7: -10.3847961, -8.7406282, -10.3694115, -8.7651777, -1.3264999, 1.3486857
8: 5.5665784, 7.2632208, 5.5951519, 7.2538681, -1.6872897, 1.6680689
9: -5.3557816, -3.8578253, -5.3400879, -3.8939795, -1.2793555, 1.2993922

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_B2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8630873, upper bound: 0.8642060
time: 7.42 seconds

## Relational analysis of NS_B2_B1_A2_A2

### Relational analysis result of NS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630879, upper bound: 0.8656554
time: 6.19 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -7.0034027, -4.3817277, -7.0502138, -4.3795738, -1.8435345, 1.8873382
1: -9.1252975, -6.9856348, -9.1307688, -6.9772491, -1.7175398, 1.7156587
2: -7.6835394, -5.9736133, -7.7209358, -5.9688120, -1.4717622, 1.4834096
3: -5.6809359, -3.6345448, -5.6883726, -3.6148036, -1.9714041, 1.9595299
4: -9.2619219, -7.2494979, -9.2687073, -7.2111216, -1.6973023, 1.6898336
5: 1.3430142, 2.7006540, 1.3402811, 2.7186661, -1.2112389, 1.1915517
6: -1.6150429, 0.3812528, -1.6832232, 0.3850598, -1.4148049, 1.4423709
7: -10.3734531, -8.7646160, -10.3848076, -8.7405872, -1.3532100, 1.3461323
8: 5.5925150, 7.2586293, 5.5665455, 7.2632251, -1.6707101, 1.6920838
9: -5.3472023, -3.8933949, -5.3557839, -3.8577628, -1.3037000, 1.3033037

Time for backsubstitution: 22.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 5829
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_B2_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653662, upper bound: 0.8642033
time: 4.74 seconds

## Relational analysis of NS_B2_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653662, upper bound: 0.8656533
time: 4.45 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -7.0113277, -4.3598185, -7.0502095, -4.3795757, -1.8524265, 1.8900013
1: -9.1286774, -6.9746399, -9.1307678, -6.9772491, -1.7215939, 1.7271247
2: -7.6889725, -5.9605584, -7.7209358, -5.9688139, -1.4878044, 1.4876184
3: -5.7180009, -3.6246285, -5.6883717, -3.6148109, -1.9820352, 1.9705925
4: -9.2679615, -7.2303629, -9.2687025, -7.2111220, -1.7035770, 1.7091365
5: 1.3210945, 2.7040460, 1.3402824, 2.7186620, -1.2157760, 1.1964397
6: -1.6205142, 0.3853800, -1.6832249, 0.3850574, -1.4218702, 1.4448161
7: -10.3841677, -8.7602415, -10.3848066, -8.7405891, -1.3590107, 1.3613129
8: 5.5846806, 7.2627721, 5.5665483, 7.2632246, -1.6785440, 1.6962237
9: -5.3533735, -3.8858504, -5.3557849, -3.8577626, -1.3099823, 1.3158481

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656507, upper bound: 0.8642013
time: 4.39 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8656507, upper bound: 0.8656512
time: 4.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.07 seconds
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8656542, upper bound: 0.8626221
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8656533, upper bound: 0.8640711
NS_B1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8642013, upper bound: 0.8643556
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8656513, upper bound: 0.8643555
NS_B2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8630873, upper bound: 0.8642060
NS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8630879, upper bound: 0.8656554
NS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8653662, upper bound: 0.8642033
NS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8653662, upper bound: 0.8656533
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8656507, upper bound: 0.8642013
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -0.8656507, upper bound: 0.8656512

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.0473433, -4.3931804, -6.9904604, -4.4055157, -1.8685904, 1.8299856
1: -9.1212082, -6.9807897, -9.1143131, -6.9904165, -1.7020388, 1.7033310
2: -7.7144804, -5.9775829, -7.6763501, -5.9787459, -1.4675884, 1.4554462
3: -5.6761150, -3.6291661, -5.6635389, -3.6464384, -1.9355078, 1.9398670
4: -9.2523222, -7.2153072, -9.2554274, -7.2520990, -1.6642056, 1.6789103
5: 1.3445467, 2.7053068, 1.3500464, 2.6897054, -1.1765842, 1.1912303
6: -1.6802418, 0.3765593, -1.6108105, 0.3743713, -1.4303036, 1.4017124
7: -10.3726339, -8.7465029, -10.3664351, -8.7718496, -1.3265429, 1.3367400
8: 5.5750999, 7.2467041, 5.6018982, 7.2490396, -1.6739397, 1.6448059
9: -5.3467922, -3.8598123, -5.3409309, -3.8950319, -1.2887521, 1.2948625

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of NS_B1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8643566, upper bound: 0.8626212
time: 4.38 seconds

## Relational analysis of NS_B1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8643575, upper bound: 0.8626221
time: 4.71 seconds

## BFS NS instance: NS_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.0483541, -4.3902612, -6.9907331, -4.4046822, -1.8711710, 1.8326893
1: -9.1268749, -6.9777608, -9.1158962, -6.9896421, -1.7079773, 1.7083097
2: -7.7184324, -5.9692273, -7.6772542, -5.9763989, -1.4771695, 1.4644828
3: -5.6810460, -3.6160264, -5.6646557, -3.6427443, -1.9447556, 1.9502239
4: -9.2677279, -7.2119255, -9.2597837, -7.2515478, -1.6713114, 1.6940184
5: 1.3417938, 2.7147775, 1.3495071, 2.6923857, -1.1822996, 1.2004943
6: -1.6823165, 0.3827996, -1.6112554, 0.3761344, -1.4366693, 1.4056497
7: -10.3841457, -8.7436171, -10.3696766, -8.7713022, -1.3334270, 1.3479135
8: 5.5702810, 7.2623520, 5.6009121, 7.2534466, -1.6831656, 1.6614399
9: -5.3538609, -3.8580260, -5.3429222, -3.8946815, -1.2920012, 1.3010383

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 5829

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 466

## Relational analysis of NS_B1_A2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8643566, upper bound: 0.8640711
time: 4.44 seconds

## Relational analysis of NS_B1_A2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8643566, upper bound: 0.8640712
time: 4.72 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -7.0483494, -4.3902626, -6.9986906, -4.3827791, -1.8736000, 1.8419180
1: -9.1268749, -6.9777613, -9.1192608, -6.9786463, -1.7197819, 1.7120223
2: -7.7184315, -5.9692268, -7.6826825, -5.9633446, -1.4816542, 1.4822721
3: -5.6810446, -3.6160312, -5.7017026, -3.6327934, -1.9518690, 1.9693346
4: -9.2677307, -7.2119246, -9.2656879, -7.2324123, -1.7003589, 1.6924057
5: 1.3417947, 2.7147763, 1.3275616, 2.6957736, -1.1861653, 1.2054667
6: -1.6823180, 0.3828011, -1.6167257, 0.3802276, -1.4368196, 1.4154396
7: -10.3841505, -8.7436199, -10.3803864, -8.7669315, -1.3540430, 1.3491635
8: 5.5702829, 7.2623506, 5.5930767, 7.2576308, -1.6873479, 1.6692739
9: -5.3538628, -3.8580251, -5.3490338, -3.8871331, -1.3088531, 1.3038168

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5829
type: A, layer: 1, pos: 466
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 4557
type: A, layer: 1, pos: 4557
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 4670

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5829

## Relational analysis of NS_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653667, upper bound: 0.8643557
time: 5.54 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653667, upper bound: 0.8640721
time: 4.25 seconds

## BFS NS instance: NS_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -7.0501051, -4.3795757, -7.0024652, -4.3908377, -1.8723660, 1.8263893
1: -9.1307631, -6.9772596, -9.1236172, -6.9862590, -1.7149601, 1.7152772
2: -7.7208848, -5.9688153, -7.6828289, -5.9806938, -1.4743032, 1.4533668
3: -5.6883669, -3.6148579, -5.6804600, -3.6361580, -1.9633598, 1.9653378
4: -9.2687006, -7.2111635, -9.2557964, -7.2503567, -1.6686268, 1.6939025
5: 1.3402836, 2.7186007, 1.3473229, 2.7010994, -1.1721697, 1.1998239
6: -1.6830862, 0.3850517, -1.6131909, 0.3668208, -1.4274268, 1.3843198
7: -10.3847923, -8.7406301, -10.3694115, -8.7651777, -1.3210764, 1.3478248
8: 5.5665789, 7.2632198, 5.5951519, 7.2538681, -1.6872892, 1.6680679
9: -5.3557806, -3.8578262, -5.3400879, -3.8939795, -1.2755051, 1.2989478

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 5829
type: B, layer: 1, pos: 4670

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630845, upper bound: 0.8653661
time: 4.31 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8630829, upper bound: 0.8656505
time: 4.41 seconds

## BFS NS instance: NS_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -7.0023909, -4.3846521, -7.0499449, -4.3803988, -1.8416681, 1.8842540
1: -9.1196442, -6.9886756, -9.1291847, -6.9780254, -1.7116213, 1.7109923
2: -7.6795831, -5.9819679, -7.7200408, -5.9711599, -1.4660215, 1.4743547
3: -5.6760077, -3.6476731, -5.6872625, -3.6184983, -1.9604406, 1.9452257
4: -9.2465096, -7.2528772, -9.2643518, -7.2116742, -1.6808949, 1.6818094
5: 1.3457384, 2.6911700, 1.3408343, 2.7159884, -1.2006130, 1.1813130
6: -1.6129642, 0.3750107, -1.6827831, 0.3832912, -1.4110117, 1.4356742
7: -10.3619194, -8.7675018, -10.3815689, -8.7411318, -1.3408484, 1.3399091
8: 5.5973248, 7.2429762, 5.5675421, 7.2588196, -1.6614947, 1.6754341
9: -5.3401251, -3.8951902, -5.3537970, -3.8581128, -1.2965837, 1.2998571

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5829
type: A, layer: 1, pos: 466
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 4557
type: B, layer: 1, pos: 4557
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5829

## Relational analysis of NS_B2_B2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.8653662, upper bound: 0.8639201
time: 4.33 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.8627997, upper bound: 0.8642030
time: 7.17 seconds

## BFS NS instance: NS_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -7.0034046, -4.3817291, -7.0502138, -4.3795738, -1.8434467, 1.8869734
1: -9.1252966, -6.9856358, -9.1307688, -6.9772491, -1.7175398, 1.7159863
2: -7.6835394, -5.9736161, -7.7209358, -5.9688120, -1.4720030, 1.4834092
3: -5.6809340, -3.6345479, -5.6883726, -3.6148036, -1.9714031, 1.9555559
4: -9.2619171, -7.2494979, -9.2687073, -7.2111216, -1.6880560, 1.6898327
5: 1.3430142, 2.7006505, 1.3402811, 2.7186661, -1.2106915, 1.1905298
6: -1.6150420, 0.3812518, -1.6832232, 0.3850598, -1.4148059, 1.4396477
7: -10.3734493, -8.7646160, -10.3848076, -8.7405872, -1.3477793, 1.3461313
8: 5.5925159, 7.2586250, 5.5665455, 7.2632251, -1.6707091, 1.6920795
9: -5.3472018, -3.8933949, -5.3557839, -3.8577628, -1.2998447, 1.3033018

Time for backsubstitution: 22.12 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.60 + 546.94 = 604.54 seconds
