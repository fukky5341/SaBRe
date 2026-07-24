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
execution time: IAR + LP analysis = 13.97 + 36.03 = 50.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.6885415, upper bound: 1.6885395


# Binary Search by BASE starts (time budget: 3550.00 seconds, max iter: 100)

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
rel_dist={7: [-1.0863711031322327, 1.086370443966616]}

## Binary Search Result
Binary search time: 207.83 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3342.17 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 945

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4391834, upper bound: 1.4371936
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4371934, upper bound: 1.4391832
time: 4.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.52
Output dim: 7, lower bound: -1.4391834, upper bound: 1.4371936
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.52
Output dim: 7, lower bound: -1.4371934, upper bound: 1.4391832

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6234035, 2.6271911
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9122577, 1.9147530
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2602577, 2.2625327
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4320791, upper bound: 1.4371838
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4391741, upper bound: 1.4302719
time: 4.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6271911, 2.6234035
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9147525, 1.9122574
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2625327, 2.2602575
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 468

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4371919, upper bound: 1.4319212
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4299487, upper bound: 1.4391817
time: 4.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 7, lower bound: -1.4320791, upper bound: 1.4371838
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 7, lower bound: -1.4391741, upper bound: 1.4302719
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 7, lower bound: -1.4371919, upper bound: 1.4319212
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.88
Output dim: 7, lower bound: -1.4299487, upper bound: 1.4391817

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5988326, 2.6187232
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8988481, 1.9101391
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2513313, 2.2366438
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4629

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4320747, upper bound: 1.4311182
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4260405, upper bound: 1.4371795
time: 3.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6149354, 2.6026204
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.9076433, 1.9013436
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2343688, 2.2536066
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4391723, upper bound: 1.4295344
time: 11.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4364883, upper bound: 1.4295356
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6053114, 2.5840421
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8908441, 1.8785496
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2341242, 2.2401175
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4302699, upper bound: 1.4319123
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4371825, upper bound: 1.4248174
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5878296, 2.6015239
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8810451, 1.8883488
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2423930, 2.2318492
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4299458, upper bound: 1.4369482
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4277556, upper bound: 1.4391793
time: 4.26 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4320747, upper bound: 1.4311182
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4260405, upper bound: 1.4371795
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4391723, upper bound: 1.4295344
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4364883, upper bound: 1.4295356
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4302699, upper bound: 1.4319123
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4371825, upper bound: 1.4248174
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4299458, upper bound: 1.4369482
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.55
Output dim: 7, lower bound: -1.4277556, upper bound: 1.4391793

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6018734, 2.6043854
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8921604, 1.8993113
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.1939144, 2.1972439
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4277307, upper bound: 1.4309574
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4319047, upper bound: 1.4270293
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5844951, 2.6217637
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8880205, 1.9034512
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2119312, 2.1792269
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4225144, upper bound: 1.4324937
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4211388, upper bound: 1.4333859
time: 4.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5682731, 2.5695648
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8887634, 1.8903332
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2427602, 2.2698803
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4354584, upper bound: 1.4295300
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4391546, upper bound: 1.4259286
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5818801, 2.5559576
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8966331, 1.8824635
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2506423, 2.2619987
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4364853, upper bound: 1.4263324
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4343004, upper bound: 1.4295308
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5807414, 2.5755742
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8774354, 1.8739359
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2251973, 2.2142279
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4302659, upper bound: 1.4319068
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4230244, upper bound: 1.4319075
time: 4.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5968442, 2.5594714
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8862307, 1.8651407
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2082348, 2.2311907
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 79

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4330594, upper bound: 1.4246475
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4370217, upper bound: 1.4204638
time: 4.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5905147, 2.6079888
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8693349, 1.8718176
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2298961, 2.2142105
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4299357, upper bound: 1.4275606
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4206518, upper bound: 1.4369389
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5942941, 2.6042094
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8645136, 1.8766387
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2247539, 2.2193530
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4241716, upper bound: 1.4391739
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4277513, upper bound: 1.4356371
time: 4.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4277307, upper bound: 1.4309574
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4319047, upper bound: 1.4270293
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4225144, upper bound: 1.4324937
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4211388, upper bound: 1.4333859
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4354584, upper bound: 1.4295300
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4391546, upper bound: 1.4259286
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4364853, upper bound: 1.4263324
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4343004, upper bound: 1.4295308
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4302659, upper bound: 1.4319068
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4230244, upper bound: 1.4319075
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4330594, upper bound: 1.4246475
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4370217, upper bound: 1.4204638
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4299357, upper bound: 1.4275606
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4206518, upper bound: 1.4369389
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4241716, upper bound: 1.4391739
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.92
Output dim: 7, lower bound: -1.4277513, upper bound: 1.4356371

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6018734, 2.6043856
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8921595, 1.8993094
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.1939135, 2.1972444
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4271959, upper bound: 1.4299562
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4267728, upper bound: 1.4303704
time: 4.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6018734, 2.6043851
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8921585, 1.8993113
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.1939144, 2.1972427
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4283894, upper bound: 1.4223210
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4270007, upper bound: 1.4231628
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5826464, 2.6210370
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8848557, 1.8989820
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2084022, 2.1742432
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4188028, upper bound: 1.4324757
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4224979, upper bound: 1.4288275
time: 4.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5837684, 2.6199155
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8835516, 1.9002864
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2069478, 2.1756978
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 468

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4133326, upper bound: 1.4333778
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4211310, upper bound: 1.4258287
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5682750, 2.5695682
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8887587, 1.8903265
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2427874, 2.2699153
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4349102, upper bound: 1.4284952
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4345535, upper bound: 1.4289405
time: 4.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5682759, 2.5695672
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8887568, 1.8903282
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2427959, 2.2699072
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4378758, upper bound: 1.4203429
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4335402, upper bound: 1.4247142
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5845661, 2.5624228
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8849232, 1.8659325
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2381468, 2.2443609
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4352094, upper bound: 1.4208152
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4309423, upper bound: 1.4251189
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5883455, 2.5586433
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8801019, 1.8707538
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2330046, 2.2495034
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 63

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4312949, upper bound: 1.4291938
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4340206, upper bound: 1.4264933
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5974827, 2.5865366
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8719554, 1.8662081
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2163649, 2.2079632
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4629

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4302615, upper bound: 1.4256557
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4242078, upper bound: 1.4319021
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5917034, 2.5923116
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8697071, 1.8684518
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2189331, 2.2053952
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4217986, upper bound: 1.4262929
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4174406, upper bound: 1.4306287
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5968442, 2.5594723
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8862307, 1.8651388
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2082338, 2.2311916
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4294169, upper bound: 1.4246396
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4330560, upper bound: 1.4209846
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5968442, 2.5594718
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8862288, 1.8651407
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2082348, 2.2311900
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4370121, upper bound: 1.4108883
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4277551, upper bound: 1.4204534
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5899463, 2.6065869
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8623216, 1.8689756
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2284088, 2.2105412
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4230035, upper bound: 1.4275517
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4299267, upper bound: 1.4194427
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5891128, 2.6074204
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8664930, 1.8648038
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2262278, 2.2127230
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4168768, upper bound: 1.4336483
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4152292, upper bound: 1.4336479
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5822067, 2.5956423
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8753386, 1.8847206
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2085004, 2.2078297
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 63

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4239050, upper bound: 1.4391502
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4239047, upper bound: 1.4357216
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.5857277, 2.5921223
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8725958, 1.8874640
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.2132320, 2.2030988
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4277507, upper bound: 1.4354063
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4272742, upper bound: 1.4356369
time: 8.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4271959, upper bound: 1.4299562
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4267728, upper bound: 1.4303704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4283894, upper bound: 1.4223210
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4270007, upper bound: 1.4231628
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4188028, upper bound: 1.4324757
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4224979, upper bound: 1.4288275
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4133326, upper bound: 1.4333778
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4211310, upper bound: 1.4258287
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4349102, upper bound: 1.4284952
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4345535, upper bound: 1.4289405
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4378758, upper bound: 1.4203429
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4335402, upper bound: 1.4247142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4352094, upper bound: 1.4208152
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4309423, upper bound: 1.4251189
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4312949, upper bound: 1.4291938
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4340206, upper bound: 1.4264933
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4302615, upper bound: 1.4256557
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4242078, upper bound: 1.4319021
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4217986, upper bound: 1.4262929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4174406, upper bound: 1.4306287
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4294169, upper bound: 1.4246396
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4330560, upper bound: 1.4209846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4370121, upper bound: 1.4108883
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4277551, upper bound: 1.4204534
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4230035, upper bound: 1.4275517
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4299267, upper bound: 1.4194427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4168768, upper bound: 1.4336483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4152292, upper bound: 1.4336479
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4239050, upper bound: 1.4391502
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4239047, upper bound: 1.4357216
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4277507, upper bound: 1.4354063
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.95
Output dim: 7, lower bound: -1.4272742, upper bound: 1.4356369

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6012921, 2.6057572
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8915982, 1.9006424
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.1937160, 2.1977072
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 468

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4259663, upper bound: 1.4244132
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4215514, upper bound: 1.4286816
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.6018734, 2.6038041
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.8921595, 1.8987477
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.6024959, 2.6024959
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.2213860, 2.2213860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7814779, 2.7814779
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0716937, 2.0716937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.1939135, 2.1970470
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4267622, upper bound: 1.4211131
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.4171937, upper bound: 1.4303592
time: 4.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 23.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.63
Output dim: 7, lower bound: -1.4259663, upper bound: 1.4244132
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.63
Output dim: 7, lower bound: -1.4215514, upper bound: 1.4286816
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 23.63
Output dim: 7, lower bound: -1.4267622, upper bound: 1.4211131
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 23.63
Output dim: 7, lower bound: -1.4171937, upper bound: 1.4303592
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4283894, upper bound: 1.4223210
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4270007, upper bound: 1.4231628
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4188028, upper bound: 1.4324757
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4224979, upper bound: 1.4288275
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4133326, upper bound: 1.4333778
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4211310, upper bound: 1.4258287
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4349102, upper bound: 1.4284952
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4345535, upper bound: 1.4289405
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4378758, upper bound: 1.4203429
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4335402, upper bound: 1.4247142
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4352094, upper bound: 1.4208152
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4309423, upper bound: 1.4251189
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4312949, upper bound: 1.4291938
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4340206, upper bound: 1.4264933
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4302615, upper bound: 1.4256557
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4242078, upper bound: 1.4319021
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4217986, upper bound: 1.4262929
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4174406, upper bound: 1.4306287
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4294169, upper bound: 1.4246396
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4330560, upper bound: 1.4209846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4370121, upper bound: 1.4108883
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4277551, upper bound: 1.4204534
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4230035, upper bound: 1.4275517
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4299267, upper bound: 1.4194427
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4168768, upper bound: 1.4336483
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4152292, upper bound: 1.4336479
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4239050, upper bound: 1.4391502
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4239047, upper bound: 1.4357216
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4277507, upper bound: 1.4354063
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.63
Output dim: 7, lower bound: -1.4272742, upper bound: 1.4356369
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.0716936588287354
rel_dist={7: [-1.4397835588004604, 1.43978321464756]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 79

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2372104, upper bound: 1.2399104
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2399106, upper bound: 1.2372097
time: 5.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.42 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.42
Output dim: 7, lower bound: -1.2372104, upper bound: 1.2399104
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.42
Output dim: 7, lower bound: -1.2399106, upper bound: 1.2372097

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4327064, 2.4327068
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7915277, 1.7915266
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4683852, 2.4683845
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1238718, 2.1238735
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7208953, 2.7208986
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0076168, 2.0076163
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0910125, 2.0910132
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353446, upper bound: 1.2355759
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2328763, upper bound: 1.2380572
time: 4.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4327068, 2.4327068
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7915268, 1.7915275
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4683847, 2.4683852
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1238737, 2.1238718
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7208982, 2.7208958
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0076158, 2.0076168
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0910134, 2.0910127
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2396163, upper bound: 1.2340507
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2367542, upper bound: 1.2369199
time: 5.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.91
Output dim: 7, lower bound: -1.2353446, upper bound: 1.2355759
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.91
Output dim: 7, lower bound: -1.2328763, upper bound: 1.2380572
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.91
Output dim: 7, lower bound: -1.2396163, upper bound: 1.2340507
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.91
Output dim: 7, lower bound: -1.2367542, upper bound: 1.2369199

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4308577, 2.4314990
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7878048, 1.7870586
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4664145, 2.4653630
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1229258, 2.1230848
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7126260, 2.7140059
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0068347, 2.0064230
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0868602, 2.0860300
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2340070, upper bound: 1.2355741
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353429, upper bound: 1.2342076
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4314985, 2.4308577
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7870591, 1.7878036
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4653640, 2.4664135
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1230831, 2.1229274
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7140021, 2.7126293
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0064237, 2.0068345
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0860291, 2.0868611
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2328719, upper bound: 1.2363983
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2312136, upper bound: 1.2380531
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4326258, 2.4326720
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7915196, 1.7915235
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4683475, 2.4682925
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1238556, 2.1238647
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7208467, 2.7208700
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0075908, 2.0075548
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0909414, 2.0909824
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2380554, upper bound: 1.2299831
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2355754, upper bound: 1.2324500
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4326720, 2.4326258
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7915225, 1.7915204
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4682918, 2.4683480
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1238670, 2.1238539
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7208734, 2.7208433
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0075545, 2.0075917
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0909834, 2.0909405
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2367513, upper bound: 1.2327122
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2325377, upper bound: 1.2369167
time: 4.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.96 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2340070, upper bound: 1.2355741
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2353429, upper bound: 1.2342076
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2328719, upper bound: 1.2363983
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2312136, upper bound: 1.2380531
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2380554, upper bound: 1.2299831
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2355754, upper bound: 1.2324500
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2367513, upper bound: 1.2327122
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.96
Output dim: 7, lower bound: -1.2325377, upper bound: 1.2369167

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4187713, 2.4214239
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7974522, 1.7951388
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4679408, 2.4666505
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1257293, 2.1296356
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7072382, 2.7075396
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0056796, 2.0054603
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0706058, 2.0724792
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2340067, upper bound: 1.2351240
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2335417, upper bound: 1.2355736
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4207830, 2.4194126
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2428913, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7958848, 1.7967062
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4677019, 2.4668899
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1294763, 2.1258881
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7061596, 2.7086182
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0058722, 2.0052676
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0733094, 2.0697756
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2304055, upper bound: 1.2342022
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2353374, upper bound: 1.2292534
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3848391, 2.3919740
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7600777, 1.7653186
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4570999, 2.4595277
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1075511, 2.1099832
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6874337, 2.6904869
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0033288, 2.0031216
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0816345, 2.0869706
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 6178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2328716, upper bound: 1.2359398
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2324210, upper bound: 1.2363979
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3926148, 2.3841987
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7645743, 1.7608218
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4584780, 2.4581497
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1101384, 2.1073956
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6918607, 2.6860600
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0027108, 2.0037401
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0861387, 2.0824664
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 468

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2312130, upper bound: 1.2351645
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2283206, upper bound: 1.2380507
time: 4.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4307775, 2.4314642
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7877963, 1.7870555
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4663768, 2.4652715
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1229095, 2.1230762
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7125773, 2.7139778
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0068088, 2.0063610
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0867882, 2.0859985
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2367212, upper bound: 1.2299812
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2380538, upper bound: 1.2285999
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4314184, 2.4308233
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7870514, 1.7878008
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4653263, 2.4663217
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1230669, 2.1229188
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7139544, 2.7126012
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0063977, 2.0067725
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0859571, 2.0868297
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 468

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2309703, upper bound: 1.2306834
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2355703, upper bound: 1.2305775
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4161820, 2.4128351
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7850776, 1.7837932
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4604864, 2.4618473
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1196284, 2.1203218
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7138500, 2.7124157
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0030448, 2.0021808
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0821486, 2.0835731
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2351661, upper bound: 1.2286721
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2326773, upper bound: 1.2311310
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4128819, 2.4161353
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7837954, 1.7850752
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4617910, 2.4605429
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1203341, 2.1196156
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7124443, 2.7138209
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0021436, 2.0030825
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0836163, 2.0821059
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2325333, upper bound: 1.2352650
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2308864, upper bound: 1.2369126
time: 4.97 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2340067, upper bound: 1.2351240
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2335417, upper bound: 1.2355736
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2304055, upper bound: 1.2342022
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2353374, upper bound: 1.2292534
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2328716, upper bound: 1.2359398
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2324210, upper bound: 1.2363979
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2312130, upper bound: 1.2351645
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2283206, upper bound: 1.2380507
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2367212, upper bound: 1.2299812
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2380538, upper bound: 1.2285999
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2309703, upper bound: 1.2306834
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2355703, upper bound: 1.2305775
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2351661, upper bound: 1.2286721
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2326773, upper bound: 1.2311310
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2325333, upper bound: 1.2352650
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.92
Output dim: 7, lower bound: -1.2308864, upper bound: 1.2369126

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4096718, 2.4186831
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2438104
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7892764, 1.7926745
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4674220, 2.4664948
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1222999, 2.1286023
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7052450, 2.7069392
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0050077, 2.0032299
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0628982, 2.0701573
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 63

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2329463, upper bound: 1.2340025
time: 4.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2329462, upper bound: 1.2327097
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4160304, 2.4123244
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7949884, 1.7869625
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4677844, 2.4661324
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1246955, 2.1262064
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7066374, 2.7055473
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0034490, 2.0047882
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0682836, 2.0647714
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2330283, upper bound: 1.2345570
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2329951, upper bound: 1.2351349
time: 8.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3962126, 2.4040442
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2121992, 2.2291362
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7824748, 1.7883222
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4492240, 2.4553246
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1230981, 2.1156936
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7021589, 2.7061071
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0003819, 2.0018330
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0571132, 2.0438859
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2285290, upper bound: 1.2341982
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2286344, upper bound: 1.2296068
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4054141, 2.3948426
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2237182, 2.2176173
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7875016, 1.7832963
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4561362, 2.4484117
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1192815, 2.1195097
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7036486, 2.7046175
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0024376, 1.9997773
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0474200, 2.0535789
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2349109, upper bound: 1.2283679
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2344658, upper bound: 1.2288115
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3757362, 2.3892293
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7519000, 1.7628534
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4565816, 2.4593716
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1041222, 2.1089497
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6854405, 2.6898861
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0026569, 2.0008912
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0739260, 2.0846474
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2308877, upper bound: 1.2359399
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2328715, upper bound: 1.2339644
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3820949, 2.3828712
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7576125, 1.7571416
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4569435, 2.4590094
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1065178, 2.1065540
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6868329, 2.6884942
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0010986, 2.0024495
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0793118, 2.0792618
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318919, upper bound: 1.2353872
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2318593, upper bound: 1.2359675
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3925338, 2.3841639
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7645671, 1.7608178
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4584417, 2.4580572
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1101203, 2.1073885
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6918073, 2.6860342
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0026855, 2.0036781
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0860658, 2.0824361
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4629

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2312091, upper bound: 1.2316096
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2276519, upper bound: 1.2351600
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3925800, 2.3841176
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7645705, 1.7608147
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4583855, 2.4581130
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1101317, 2.1073775
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6918340, 2.6860075
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0026484, 2.0037150
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0861077, 2.0823936
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2269343, upper bound: 1.2380488
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2283189, upper bound: 1.2367162
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4186912, 2.4213891
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7974436, 1.7951355
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4679041, 2.4665594
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1257131, 2.1296268
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7071886, 2.7075109
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0056539, 2.0053988
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0705347, 2.0724485
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 6178

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2321306, upper bound: 1.2281757
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2367162, upper bound: 1.2280698
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4207025, 2.4193778
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2428844, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7958767, 1.7967029
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4676647, 2.4667988
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1294601, 2.1258795
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7061110, 2.7085891
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0058465, 2.0052061
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0732384, 2.0697448
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2380474, upper bound: 1.2267440
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2336691, upper bound: 1.2267592
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4190564, 2.4111252
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7780058, 1.7821183
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4648027, 2.4654882
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1130669, 2.1069751
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7078862, 2.7087903
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9912887, 1.9972901
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0771947, 2.0813303
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2309633, upper bound: 1.2278310
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2252241, upper bound: 1.2278453
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4117203, 2.4184632
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7813709, 1.7787554
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4644928, 2.4657984
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1071236, 2.1129220
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7101436, 2.7065334
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9969163, 1.9916637
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0804577, 2.0780673
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2355633, upper bound: 1.2277255
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2298288, upper bound: 1.2277398
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4143338, 2.4116282
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7813542, 1.7793248
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4585156, 2.4588263
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1186826, 2.1195331
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7055812, 2.7055225
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0022621, 2.0009866
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0779967, 2.0785899
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2305743, upper bound: 1.2268635
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2351611, upper bound: 1.2267574
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4149747, 2.4109874
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7806089, 1.7800701
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4574656, 2.4598765
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1188395, 2.1193757
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7069583, 2.7041464
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0018506, 2.0013978
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0771656, 2.0794210
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2313093, upper bound: 1.2311289
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2326755, upper bound: 1.2298012
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3662224, 2.3772516
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7568140, 1.7625906
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4535279, 2.4536581
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1048014, 2.1066701
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6858740, 2.6916771
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9990492, 1.9993699
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0792217, 2.0822151
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2321073, upper bound: 1.2343955
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2316643, upper bound: 1.2348383
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3739977, 2.3694758
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2441764
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7613106, 1.7580938
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4549060, 2.4522800
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1073887, 2.1040828
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6903009, 2.6872501
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9984312, 1.9999883
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0837259, 2.0777113
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2308780, upper bound: 1.2327587
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2267281, upper bound: 1.2369060
time: 4.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2329463, upper bound: 1.2340025
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2329462, upper bound: 1.2327097
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2330283, upper bound: 1.2345570
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2329951, upper bound: 1.2351349
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2285290, upper bound: 1.2341982
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2286344, upper bound: 1.2296068
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2349109, upper bound: 1.2283679
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2344658, upper bound: 1.2288115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2308877, upper bound: 1.2359399
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2328715, upper bound: 1.2339644
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2318919, upper bound: 1.2353872
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2318593, upper bound: 1.2359675
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2312091, upper bound: 1.2316096
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2276519, upper bound: 1.2351600
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2269343, upper bound: 1.2380488
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2283189, upper bound: 1.2367162
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2321306, upper bound: 1.2281757
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2367162, upper bound: 1.2280698
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2380474, upper bound: 1.2267440
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2336691, upper bound: 1.2267592
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2309633, upper bound: 1.2278310
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2252241, upper bound: 1.2278453
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2355633, upper bound: 1.2277255
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2298288, upper bound: 1.2277398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2305743, upper bound: 1.2268635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2351611, upper bound: 1.2267574
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2313093, upper bound: 1.2311289
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2326755, upper bound: 1.2298012
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2321073, upper bound: 1.2343955
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2316643, upper bound: 1.2348383
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2308780, upper bound: 1.2327587
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.18
Output dim: 7, lower bound: -1.2267281, upper bound: 1.2369060

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4096427, 2.4186587
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2438204
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7892833, 1.7926800
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4674678, 2.4665296
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1223254, 2.1286352
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7052875, 2.7069888
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0050459, 2.0032611
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0628815, 2.0701435
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2280091, upper bound: 1.2339966
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2329408, upper bound: 1.2290684
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.4096446, 2.4186535
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2441764, 2.2438188
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7892814, 1.7926800
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4674573, 2.4665396
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.1223264, 2.1286280
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.7052751, 2.7069807
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -2.0050387, 2.0032520
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0628805, 2.0701406
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.13 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.0076169967651367
rel_dist={7: [-1.2399111761101596, 1.2399105900004943]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6178
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 61

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6178

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1600642, upper bound: 1.1637595
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637599, upper bound: 1.1600635
time: 7.14 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.34 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.34
Output dim: 7, lower bound: -1.1600642, upper bound: 1.1637595
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.34
Output dim: 7, lower bound: -1.1637599, upper bound: 1.1600635

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3402414, 2.3471427
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1946495, 2.2032888
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7350249, 1.7387946
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3913875, 2.3965721
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0560641, 2.0532019
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6637082, 2.6648250
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9685390, 1.9700806
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0133805, 2.0061109
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 175

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1597631, upper bound: 1.1630119
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1593133, upper bound: 1.1634629
time: 8.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3471427, 2.3402414
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.2032888, 2.1946497
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7387948, 1.7350252
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3965721, 2.3913872
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0532017, 2.0560639
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6648250, 2.6637077
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9700806, 1.9685390
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0061107, 2.0133803
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637551, upper bound: 1.1558240
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1595190, upper bound: 1.1600585
time: 4.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.01
Output dim: 7, lower bound: -1.1597631, upper bound: 1.1630119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.01
Output dim: 7, lower bound: -1.1593133, upper bound: 1.1634629
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.01
Output dim: 7, lower bound: -1.1637551, upper bound: 1.1558240
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.01
Output dim: 7, lower bound: -1.1595190, upper bound: 1.1600585

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3396597, 2.3473988
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1942594, 2.2034469
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7344637, 1.7390459
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3913937, 2.3965521
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0555191, 2.0534520
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6633286, 2.6649971
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9686954, 1.9697301
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0131836, 2.0061965
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 484

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 61

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1597627, upper bound: 1.1608678
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1576239, upper bound: 1.1630110
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3402414, 2.3465610
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1946495, 2.2028985
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7350249, 1.7382331
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3913674, 2.3965721
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0560641, 2.0526571
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6637082, 2.6644459
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9681885, 1.9700806
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0133805, 2.0059137
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5859

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1593107, upper bound: 1.1622327
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1580823, upper bound: 1.1634599
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3460975, 2.3388391
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1972549, 2.1901529
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7317805, 1.7297988
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3947897, 2.3889954
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0469227, 2.0513854
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6628690, 2.6610823
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9640357, 1.9604254
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0033770, 2.0097115
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1627914, upper bound: 1.1558229
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637538, upper bound: 1.1548602
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3457403, 2.3391962
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1987922, 2.1886156
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7335677, 1.7280109
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3941803, 2.3896050
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0485229, 2.0497849
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6621995, 2.6617517
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9619672, 1.9624937
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0024414, 2.0106461
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1582094, upper bound: 1.1565258
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1560101, upper bound: 1.1587367
time: 4.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.11 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1597627, upper bound: 1.1608678
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1576239, upper bound: 1.1630110
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1593107, upper bound: 1.1622327
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1580823, upper bound: 1.1634599
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1627914, upper bound: 1.1558229
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1637538, upper bound: 1.1548602
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1582094, upper bound: 1.1565258
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 23.11
Output dim: 7, lower bound: -1.1560101, upper bound: 1.1587367

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3395786, 2.3473525
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1942506, 2.2034359
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7344565, 1.7390411
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3913426, 2.3964596
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0555010, 2.0534420
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6632771, 2.6649656
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9686608, 1.9696684
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0131111, 2.0061560
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 13.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 63

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1580421, upper bound: 1.1596007
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1584987, upper bound: 1.1591438
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3396134, 2.3473177
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1942487, 2.2034383
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7344584, 1.7390387
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3913012, 2.3965011
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0555091, 2.0534339
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6632962, 2.6649456
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9686332, 1.9696960
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0131426, 2.0061240
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1576214, upper bound: 1.1597884
time: 8.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1543994, upper bound: 1.1630086
time: 8.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2935815, 2.3057308
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1823096, 2.1920948
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7080436, 1.7146196
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3831029, 2.3893414
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0405316, 2.0390618
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6371355, 2.6411953
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9649401, 1.9663682
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0089865, 2.0048959
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7339106

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1583361, upper bound: 1.1622318
time: 7.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1593094, upper bound: 1.1612678
time: 8.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2994132, 2.2998991
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1838484, 2.1905560
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7114162, 1.7112470
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3841367, 2.3883080
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0424724, 2.0371213
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6404562, 2.6378751
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9644766, 1.9668322
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0123644, 2.0015180
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7351892

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1580802, upper bound: 1.1603051
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1549307, upper bound: 1.1634577
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3340111, 2.3282614
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1790941, 2.1693947
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7410369, 1.7378798
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3962574, 2.3902841
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0497270, 2.0570002
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6572094, 2.6546144
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9628811, 1.9594157
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9871225, 1.9954853
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4629

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1627877, upper bound: 1.1531706
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1601417, upper bound: 1.1558187
time: 9.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3355198, 2.3267527
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1764967, 2.1719921
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7398610, 1.7390554
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3960781, 2.3904634
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0525374, 2.0541897
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6564007, 2.6554232
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9630260, 1.9592712
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9891505, 1.9934576
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637513, upper bound: 1.1516392
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1605314, upper bound: 1.1548575
time: 11.21 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 31.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1580421, upper bound: 1.1596007
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1584987, upper bound: 1.1591438
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1576214, upper bound: 1.1597884
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1543994, upper bound: 1.1630086
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1583361, upper bound: 1.1622318
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1593094, upper bound: 1.1612678
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1580802, upper bound: 1.1603051
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1549307, upper bound: 1.1634577
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1627877, upper bound: 1.1531706
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1601417, upper bound: 1.1558187
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1637513, upper bound: 1.1516392
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.06
Output dim: 7, lower bound: -1.1605314, upper bound: 1.1548575

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3395500, 2.3473253
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1942587, 2.2034450
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7344522, 1.7390368
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3913846, 2.3964939
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0554571, 2.0533986
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6633177, 2.6650128
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9686983, 1.9697001
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0130949, 2.0061393
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1575437, upper bound: 1.1596019
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1575767, upper bound: 1.1586463
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3077450, 2.3079567
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1866658, 2.1968014
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7049501, 1.7053304
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3722644, 2.3813212
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0401924, 2.0400298
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6479931, 2.6474547
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9524784, 1.9512358
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9847336, 1.9812586
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1555144, upper bound: 1.1555326
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1555037, upper bound: 1.1597835
time: 8.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3002524, 2.3154492
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1876113, 2.1958554
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7007501, 1.7095301
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3761206, 2.3774650
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0421054, 2.0381169
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6458054, 2.6496415
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9501729, 1.9535415
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9882774, 1.9777150
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1543955, upper bound: 1.1597912
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1543952, upper bound: 1.1630063
time: 7.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2814946, 2.2951536
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1641498, 2.1713383
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7172999, 1.7227006
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3845711, 2.3906291
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0433352, 2.0446754
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6314793, 2.6347294
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9637861, 1.9653590
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9927316, 1.9906693
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7357051

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 79

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1583306, upper bound: 1.1588059
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1549012, upper bound: 1.1622262
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2830033, 2.2936454
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1615524, 2.1739357
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7161241, 1.7238762
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3843918, 2.3908086
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0461457, 2.0418649
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6306705, 2.6355381
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9639301, 1.9652145
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9947600, 1.9886415
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7359552

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 466

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1591752, upper bound: 1.1607081
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1591761, upper bound: 1.1612685
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2820983, 2.2801094
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1824858, 2.1893618
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7046509, 1.7035203
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3763323, 2.3814809
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0382340, 2.0334127
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6330824, 2.6294460
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9597421, 1.9614213
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0035300, 1.9937844
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7351868

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 63

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1563601, upper bound: 1.1590419
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1568168, upper bound: 1.1585845
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2796235, 2.2825847
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1826537, 2.1891935
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7036896, 1.7044818
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3773108, 2.3805029
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0387633, 2.0328829
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6320276, 2.6305003
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9590659, 1.9620974
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -2.0046306, 1.9926839
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7353516

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1548559, upper bound: 1.1597817
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1548510, upper bound: 1.1634549
time: 7.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3271222, 2.3139243
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1752903, 2.1650474
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7319832, 1.7270517
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3995471, 2.3958294
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0664301, 2.0767326
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6269951, 2.6200676
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9460979, 1.9397688
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9297056, 1.9457901
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1614785, upper bound: 1.1496560
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1592599, upper bound: 1.1518626
time: 4.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3196740, 2.3213725
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1747468, 2.1655910
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7302089, 1.7288260
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.4018030, 2.3935738
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0694594, 2.0737033
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6226625, 2.6243997
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9432344, 1.9426322
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9374275, 1.9380686
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1601414, upper bound: 1.1554812
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1598065, upper bound: 1.1558190
time: 5.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3036504, 2.2873912
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1689153, 2.1653566
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7103529, 1.7053475
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3770418, 2.3752835
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0372214, 2.0407860
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6410966, 2.6379323
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9468703, 1.9408097
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9607410, 1.9685919
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 945

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1637492, upper bound: 1.1516349
time: 8.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1605321, upper bound: 1.1516351
time: 6.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2961578, 2.2948837
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1698608, 2.1644106
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7061539, 1.7095473
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3808985, 2.3714273
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0391345, 2.0388734
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6389098, 2.6401191
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9445643, 1.9431157
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9642849, 1.9650483
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1592120, upper bound: 1.1513440
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1569908, upper bound: 1.1535555
time: 4.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1575437, upper bound: 1.1596019
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1575767, upper bound: 1.1586463
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1555144, upper bound: 1.1555326
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1555037, upper bound: 1.1597835
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1543955, upper bound: 1.1597912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1543952, upper bound: 1.1630063
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1583306, upper bound: 1.1588059
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1549012, upper bound: 1.1622262
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1591752, upper bound: 1.1607081
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1591761, upper bound: 1.1612685
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1563601, upper bound: 1.1590419
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1568168, upper bound: 1.1585845
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1548559, upper bound: 1.1597817
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1548510, upper bound: 1.1634549
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1614785, upper bound: 1.1496560
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1592599, upper bound: 1.1518626
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1601414, upper bound: 1.1554812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1598065, upper bound: 1.1558190
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1637492, upper bound: 1.1516349
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1605321, upper bound: 1.1516351
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1592120, upper bound: 1.1513440
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 23.08
Output dim: 7, lower bound: -1.1569908, upper bound: 1.1535555

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3274622, 2.3367481
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1760988, 2.1826878
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7437184, 1.7471266
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3928523, 2.3977823
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0583296, 2.0590868
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6576605, 2.6585469
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9675436, 1.9686904
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9968414, 1.9919164
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 484

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1540720, upper bound: 1.1595948
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1575393, upper bound: 1.1561271
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3063426, 2.3069115
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1821702, 2.1907685
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.6997259, 1.6983171
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3698740, 2.3795400
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0355175, 2.0337548
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6453667, 2.6454973
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9443641, 1.9451900
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9810643, 1.9785244
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1534237, upper bound: 1.1597830
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1555033, upper bound: 1.1576933
time: 7.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3136892, 2.3264105
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1862490, 2.1946609
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.6939843, 1.7018023
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3804569, 2.3827798
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0386958, 2.0352373
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6384325, 2.6412139
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9454381, 1.9481311
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9794450, 1.9699836
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 63

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 468

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1534255, upper bound: 1.1597876
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1543942, upper bound: 1.1588279
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.3112144, 2.3288872
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1864188, 2.1944926
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.6930230, 1.7027659
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3814354, 2.3818016
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0392256, 2.0347078
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6373787, 2.6422677
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9447620, 1.9488070
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9805455, 1.9688830
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7373474

Time for backsubstitution: 14.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 5746
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 468
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 5859
type: RSZ, layer: 1, pos: 4629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 466

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1522899, upper bound: 1.1587506
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1522792, upper bound: 1.1630017
time: 6.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2858000, 2.2978392
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1318235, 2.1344032
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7007689, 1.7082367
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3781905, 2.3872037
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4360805
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0339692, 2.0339739
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6216736, 2.6205215
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9605768, 1.9636397
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9750938, 1.9752350
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7255576, 1.7201385

Time for backsubstitution: 14.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 79
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 5805
type: RSZ, layer: 1, pos: 484
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 63
type: RSZ, layer: 1, pos: 466
type: RSZ, layer: 1, pos: 61
type: RSZ, layer: 1, pos: 4629
type: RSZ, layer: 1, pos: 580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 79

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1528148, upper bound: 1.1622257
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1549008, upper bound: 1.1601467
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.1188126, -4.3942251, -7.1188126, -4.3942251, -2.2739010, 2.2893121
1: -7.3048649, -5.0606885, -7.3048649, -5.0606885, -2.1610579, 2.1728992
2: -6.1131477, -4.0248523, -6.1131477, -4.0248523, -1.7079463, 1.7199821
3: -6.1664658, -3.5639699, -6.1664658, -3.5639699, -2.3838739, 2.3905621
4: -6.4951239, -4.0573511, -6.4951239, -4.0573511, -2.4377728, 2.4377728
5: -6.5228052, -4.3014193, -6.5228052, -4.3014193, -2.0427163, 2.0402334
6: -11.4839764, -8.7024984, -11.4839764, -8.7024984, -2.6286783, 2.6345897
7: 2.7477748, 4.8194685, 2.7477748, 4.8194685, -1.9628692, 1.9629831
8: -4.4071116, -2.0474048, -4.4071116, -2.0474048, -1.9870510, 1.9849722
9: -2.7929399, -1.0555925, -2.7929399, -1.0555925, -1.7373474, 1.7342434

Time for backsubstitution: 14.09 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.974029779434204
rel_dist={7: [-1.163764129436149, 1.1637638135586528]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.79 seconds
