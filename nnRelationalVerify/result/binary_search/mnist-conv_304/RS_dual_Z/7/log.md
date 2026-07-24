## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.9085582323
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.4833460, 2.4833460)
1: (-19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.7038298, 3.7038298)
2: (-6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.8991394, 2.8991392)
3: (-10.8192272, -7.7928076, -10.8192272, -7.7928076, -3.0264196, 3.0264196)
4: (-13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.9983845, 2.9983845)
5: (-4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.4810138, 2.4810138)
6: (-4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.5990338, 2.5990338)
7: (-12.8235607, -8.7824364, -12.8235607, -8.7824364, -4.0411243, 4.0411243)
8: (-5.4501801, -3.1462440, -5.4501801, -3.1462440, -2.2276306, 2.2276306)
9: (-1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725)

## BASE Result
execution time: IAR + LP analysis = 14.02 + 33.53 = 47.54 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.46 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2897982597351074
rel_dist={0: [-1.319613808150022, 1.3196135105828688]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0339293479919434
rel_dist={0: [-0.9094671207244254, 0.9094665870331546]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8633503913879395
rel_dist={0: [-0.5847592780895585, 0.5847577424659303]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.9486398696899414
rel_dist={0: [-0.7535712826074459, 0.7535698485994864]}

## Binary Search Result
Binary search time: 211.02 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3341.44 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4289859, upper bound: 1.4297160
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297157, upper bound: 1.4289859
time: 5.36 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.83
Output dim: 0, lower bound: -1.4289859, upper bound: 1.4297160
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.83
Output dim: 0, lower bound: -1.4297157, upper bound: 1.4289859

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3743129, 2.3831048
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0035734, 2.9998355
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3597670, 2.3637965
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7925549, 2.7901568
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7188334, 2.7117033
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0756845, 2.0725861
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3882895, 2.3933029
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5252571, 3.5215511
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7933290, 1.7918472
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4286474, upper bound: 1.4297107
time: 8.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4289791, upper bound: 1.4293769
time: 21.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3750882, 2.3743129
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9998350, 3.0001650
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3601246, 2.3597672
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7901573, 2.7903666
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7117038, 2.7123322
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0725861, 2.0728593
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3887277, 2.3882890
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5215511, 3.5218782
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7918475, 1.7919779
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4293772, upper bound: 1.4289791
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297088, upper bound: 1.4286473
time: 5.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 0, lower bound: -1.4286474, upper bound: 1.4297107
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 0, lower bound: -1.4289791, upper bound: 1.4293769
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 0, lower bound: -1.4293772, upper bound: 1.4289791
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.42
Output dim: 0, lower bound: -1.4297088, upper bound: 1.4286473

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3770423, 2.3822737
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0016394, 3.0061865
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3640218, 2.3625033
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7928391, 2.7900710
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7198505, 2.7113934
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0744820, 2.0765352
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3860669, 2.4006014
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5394487, 3.5172262
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7924664, 1.7946780
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4286415, upper bound: 1.4224620
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4213987, upper bound: 1.4297027
time: 8.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3734813, 2.3831048
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0035734, 2.9979014
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3584743, 2.3637965
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7924690, 2.7901568
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7185240, 2.7117033
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0756845, 2.0713832
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3882895, 2.3910804
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5209322, 3.5215511
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7933290, 1.7909849
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4289732, upper bound: 1.4221286
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4217303, upper bound: 1.4293713
time: 7.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3778176, 2.3734818
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9979010, 3.0065160
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3643785, 2.3584745
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7904396, 2.7902803
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7127199, 2.7120223
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0713830, 2.0768087
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3865056, 2.3955879
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5357437, 3.5175543
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7909844, 1.7948086
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4293713, upper bound: 1.4217298
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4221285, upper bound: 1.4289751
time: 6.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3742566, 2.3743129
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9998350, 2.9982309
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3588309, 2.3597672
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7900715, 2.7903666
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7113934, 2.7123322
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0725861, 2.0716567
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3887277, 2.3860664
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5172262, 3.5218782
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7918475, 1.7911155
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297030, upper bound: 1.4213982
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4224601, upper bound: 1.4286434
time: 6.26 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4286415, upper bound: 1.4224620
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4213987, upper bound: 1.4297027
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4289732, upper bound: 1.4221286
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4217303, upper bound: 1.4293713
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4293713, upper bound: 1.4217298
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4221285, upper bound: 1.4289751
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4297030, upper bound: 1.4213982
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.19
Output dim: 0, lower bound: -1.4224601, upper bound: 1.4286434

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3670640, 2.3679459
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9864416, 2.9954185
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3645620, 2.3625016
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8015256, 2.7961969
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7019563, 2.6994014
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0692754, 2.0744154
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3478889, 2.3466930
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5238895, 3.4917698
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7758634, 1.7712314
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9658728, 2.9649482

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4286217, upper bound: 1.4136956
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198792, upper bound: 1.4224407
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3627148, 2.3722949
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9908705, 2.9909887
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3640203, 2.3630428
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7989650, 2.7987576
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7078576, 2.6935000
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0723619, 2.0713286
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3321581, 2.3624239
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5139923, 3.5016661
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7690198, 1.7780745
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9732275, 2.9575930

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4213789, upper bound: 1.4209426
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4126344, upper bound: 1.4296839
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3635030, 2.3687770
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9883747, 2.9871330
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3590136, 2.3637953
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8011565, 2.7962837
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7006307, 2.6997113
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0704780, 2.0692635
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3501110, 2.3371720
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5053720, 3.4960942
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7767260, 1.7675381
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9635849, 2.9654827

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4289534, upper bound: 1.4133644
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4202109, upper bound: 1.4221091
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3591542, 2.3731263
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9928045, 2.9827042
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3584728, 2.3643370
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7985950, 2.7988443
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7065320, 2.6938095
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0735645, 2.0661764
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3343801, 2.3529024
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4954758, 3.5059910
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7698829, 1.7743814
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9709396, 2.9581275

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4217105, upper bound: 1.4206086
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4129660, upper bound: 1.4293523
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3678389, 2.3591540
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9827042, 2.9957480
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3649178, 2.3584728
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7991271, 2.7964063
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6948266, 2.7000318
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0661764, 2.0746889
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3483272, 2.3416796
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5201836, 3.4920974
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7743814, 1.7713621
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9598808, 2.9654727

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4293521, upper bound: 1.4129657
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4206091, upper bound: 1.4217100
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3634901, 2.3635030
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9871340, 2.9913182
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3643761, 2.3590140
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7965665, 2.7989669
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7007279, 2.6941299
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0692635, 2.0716019
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3325963, 2.3574100
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5102863, 3.5019941
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7675378, 1.7782054
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9672365, 2.9581180

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4221093, upper bound: 1.4202111
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4133643, upper bound: 1.4289540
time: 5.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3642783, 2.3599851
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9846382, 2.9874625
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3593702, 2.3597670
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7987571, 2.7964931
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6934991, 2.7003412
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0673790, 2.0695367
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3505497, 2.3321581
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5016661, 3.4964209
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7752445, 1.7676687
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9575930, 2.9660072

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4296838, upper bound: 1.4126342
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4209407, upper bound: 1.4213784
time: 5.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3599291, 2.3643343
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9890671, 2.9830337
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3588285, 2.3603082
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7961965, 2.7990537
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6994014, 2.6944399
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0704660, 2.0664501
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3348188, 2.3478889
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4917698, 3.5063171
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7684009, 1.7745121
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9649477, 2.9586520

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4224409, upper bound: 1.4198794
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4136959, upper bound: 1.4286220
time: 6.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 31.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4286217, upper bound: 1.4136956
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4198792, upper bound: 1.4224407
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4213789, upper bound: 1.4209426
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4126344, upper bound: 1.4296839
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4289534, upper bound: 1.4133644
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4202109, upper bound: 1.4221091
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4217105, upper bound: 1.4206086
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4129660, upper bound: 1.4293523
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4293521, upper bound: 1.4129657
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4206091, upper bound: 1.4217100
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4221093, upper bound: 1.4202111
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4133643, upper bound: 1.4289540
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4296838, upper bound: 1.4126342
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4209407, upper bound: 1.4213784
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4224409, upper bound: 1.4198794
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 0, lower bound: -1.4136959, upper bound: 1.4286220

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3488445, 2.3422496
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9816418, 2.9927959
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3570375, 2.3571744
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8129411, 2.8040004
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6854196, 2.6899867
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0719042, 2.0782547
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3412452, 2.3350520
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5009613, 3.4761930
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7679477, 1.7600522
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9610615, 2.9615421

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4286155, upper bound: 1.4132070
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4208845, upper bound: 1.4132155
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3413677, 2.3497262
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9838171, 2.9906187
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3592329, 2.3549771
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8093295, 2.8076115
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6925416, 2.6828637
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0731144, 2.0770445
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3362479, 2.3400502
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5083122, 3.4688416
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7646842, 1.7633159
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9624653, 2.9601374

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198730, upper bound: 1.4219441
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4121468, upper bound: 1.4219525
time: 17.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3444953, 2.3465986
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9860706, 2.9883671
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3564959, 2.3577156
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8103795, 2.8065610
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6913209, 2.6840849
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0749912, 2.0751677
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3255148, 2.3507829
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4910641, 3.4860897
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7611041, 1.7668955
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9684172, 2.9541874

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4208915, upper bound: 1.4132083
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4208831, upper bound: 1.4209350
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3370185, 2.3540754
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9882469, 2.9861894
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3586912, 2.3555183
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8067689, 2.8101721
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6984429, 2.6769619
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0762014, 2.0739577
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3205171, 2.3557811
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4984159, 3.4787383
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7578406, 1.7701590
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9698200, 2.9527822

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4121538, upper bound: 1.4219451
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4121454, upper bound: 1.4296780
time: 6.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3452835, 2.3430822
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9835758, 2.9845114
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3514891, 2.3584676
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8125710, 2.8040876
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6840920, 2.6902962
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0731063, 2.0731025
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3434677, 2.3255310
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4824438, 3.4805174
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7688107, 1.7563591
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9587736, 2.9620762

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4289472, upper bound: 1.4128748
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4212149, upper bound: 1.4128838
time: 5.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3378067, 2.3505588
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9857502, 2.9823337
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3536844, 2.3562708
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8089595, 2.8076982
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6912160, 2.6831732
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0743165, 2.0718925
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3384700, 2.3305292
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4897957, 3.4731655
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7655473, 1.7596226
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9601765, 2.9606714

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4202046, upper bound: 1.4216129
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4124785, upper bound: 1.4216222
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3409343, 2.3474312
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9880047, 2.9800820
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3509483, 2.3590088
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8100095, 2.8066483
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6899943, 2.6843944
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0761933, 2.0700157
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3277369, 2.3412619
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4725475, 3.4904137
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7619672, 1.7632024
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9661283, 2.9547215

Time for backsubstitution: 14.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4212219, upper bound: 1.4128768
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4212135, upper bound: 1.4206036
time: 6.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3334579, 2.3549080
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9901791, 2.9779043
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3531437, 2.3568115
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8063989, 2.8102593
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6971173, 2.6772714
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0774035, 2.0688057
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3227391, 2.3462601
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4798985, 3.4830627
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7587037, 1.7664659
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9675322, 2.9533167

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4124855, upper bound: 1.4216154
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4124771, upper bound: 1.4293466
time: 6.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3496194, 2.3334577
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9779043, 2.9931259
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3573923, 2.3531437
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8105416, 2.8042097
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6782880, 2.6906166
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0688057, 2.0785277
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3416834, 2.3300381
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4972553, 3.4765205
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7664657, 1.7601831
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9550705, 2.9620667

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4293459, upper bound: 1.4124771
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4216147, upper bound: 1.4124860
time: 5.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3421426, 2.3409343
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9800816, 2.9909487
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3595886, 2.3509483
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8069310, 2.8078208
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6854119, 2.6834936
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0700154, 2.0773177
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3366857, 2.3350363
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5046072, 3.4691691
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7632022, 1.7634468
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9564753, 2.9606619

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4206029, upper bound: 1.4212135
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4128764, upper bound: 1.4212224
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3452697, 2.3378067
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9823341, 2.9886966
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3568516, 2.3536849
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8079810, 2.8067703
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6841903, 2.6847148
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0718923, 2.0754409
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3259525, 2.3457689
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4873590, 3.4864168
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7596231, 1.7670264
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9624252, 2.9547119

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4216216, upper bound: 1.4124805
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4216133, upper bound: 1.4202051
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3377934, 2.3452835
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9845114, 2.9865193
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3590469, 2.3514895
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8043704, 2.8103814
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6913133, 2.6775918
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0731025, 2.0742307
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3209548, 2.3507667
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4947100, 3.4790659
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7563586, 1.7702901
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9638300, 2.9533067

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4128834, upper bound: 1.4212155
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4128750, upper bound: 1.4289476
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3460584, 2.3342903
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9798374, 2.9848409
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3518448, 2.3544374
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8101726, 2.8042970
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6769624, 2.6909266
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0700078, 2.0733757
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3439064, 2.3205171
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4787388, 3.4808445
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7673287, 1.7564900
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9527826, 2.9626026

Time for backsubstitution: 14.64 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.3750882148742676
rel_dist={0: [-1.4297502739106367, 1.4297521247570089]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0566408, upper bound: 1.0573698
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573696, upper bound: 1.0566410
time: 6.07 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.99
Output dim: 0, lower bound: -1.0566408, upper bound: 1.0573698
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.99
Output dim: 0, lower bound: -1.0573696, upper bound: 1.0566410

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1184440, 2.1234679
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5797720, 2.5776367
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0363579, 2.0386603
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4709072, 2.4695358
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3375978, 2.3335233
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8080497, 1.8062789
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1511574, 2.1540222
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.1061440, 3.1040258
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5313027, 1.5304558
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7730103, 2.7695866

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565757, upper bound: 1.0573680
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0566394, upper bound: 1.0573026
time: 8.51 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1192193, 2.1184440
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5776377, 2.5779662
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0367146, 2.0363581
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4695358, 2.4697452
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3335233, 2.3341522
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8062787, 1.8065524
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1515956, 2.1511569
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.1040258, 3.1043530
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5304558, 1.5305865
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7695866, 2.7701116

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573027, upper bound: 1.0566396
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573681, upper bound: 1.0565738
time: 7.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 28.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 0, lower bound: -1.0565757, upper bound: 1.0573680
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 0, lower bound: -1.0566394, upper bound: 1.0573026
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 0, lower bound: -1.0573027, upper bound: 1.0566396
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 28.51
Output dim: 0, lower bound: -1.0573681, upper bound: 1.0565738

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1196475, 2.1226368
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5778379, 2.5804372
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0382352, 2.0373676
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4710312, 2.4694500
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3380461, 2.3332133
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8068471, 1.8080204
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1489348, 2.1572404
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.1124001, 3.0997009
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5304396, 1.5317037
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7737837, 2.7690525

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565707, upper bound: 1.0531859
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523917, upper bound: 1.0573647
time: 6.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1176124, 2.1234679
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5797720, 2.5757031
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0350652, 2.0386603
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4708195, 2.4695358
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3372879, 2.3335233
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8080497, 1.8050761
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1511574, 2.1517997
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.1018181, 3.1040258
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5313027, 1.5295932
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7724762, 2.7695866

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0566362, upper bound: 1.0531199
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524573, upper bound: 1.0572990
time: 5.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1204224, 2.1176128
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5757036, 2.5807667
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0385919, 2.0350654
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4696617, 2.4696589
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3339715, 2.3338428
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8050761, 1.8082938
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1493735, 2.1543756
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.1102819, 3.1000290
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5295928, 1.5318344
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7703600, 2.7695780

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0572994, upper bound: 1.0524579
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531199, upper bound: 1.0566359
time: 6.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1183877, 2.1184440
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5776377, 2.5760326
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0354218, 2.0363581
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4694500, 2.4697452
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3332133, 2.3341522
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8062787, 1.8053498
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1515956, 2.1489348
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0997009, 3.1043530
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5304558, 1.5297239
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7690525, 2.7701116

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573649, upper bound: 1.0523917
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531852, upper bound: 1.0565708
time: 6.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0565707, upper bound: 1.0531859
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0523917, upper bound: 1.0573647
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0566362, upper bound: 1.0531199
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0524573, upper bound: 1.0572990
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0572994, upper bound: 1.0524579
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0531199, upper bound: 1.0566359
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0573649, upper bound: 1.0523917
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.23
Output dim: 0, lower bound: -1.0531852, upper bound: 1.0565708

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1078048, 2.1083088
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5626411, 2.5677710
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0385427, 2.0373659
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4786210, 2.4755759
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3201518, 2.3186922
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8016405, 1.8045776
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1040149, 2.1033316
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0925980, 3.0742445
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5109041, 1.5082572
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7485323, 2.7480040

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565600, upper bound: 1.0482600
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516430, upper bound: 1.0531752
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1053195, 2.1107941
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5651722, 2.5652399
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0382338, 2.0376754
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4771571, 2.4770393
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3235245, 2.3153195
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8034043, 1.8028135
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0950265, 2.1123209
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0869427, 3.0798998
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5069931, 1.5121675
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7527351, 2.7438011

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523810, upper bound: 1.0524387
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474638, upper bound: 1.0573539
time: 6.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1057701, 2.1091404
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5645752, 2.5630364
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0353727, 2.0386600
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4784093, 2.4756627
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3193941, 2.3190022
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8028431, 1.8016334
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1062369, 2.0978913
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0820169, 3.0785685
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5117667, 1.5061467
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7472248, 2.7485380

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0566255, upper bound: 1.0481933
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517089, upper bound: 1.0531093
time: 5.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1032848, 2.1116257
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5671062, 2.5605054
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0350637, 2.0389690
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4769464, 2.4771261
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3227668, 2.3156300
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8046069, 1.7998695
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0972486, 2.1068802
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0763617, 3.0842242
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5078561, 1.5100572
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7514277, 2.7443352

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524466, upper bound: 1.0523730
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0475323, upper bound: 1.0572888
time: 6.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1085801, 2.1032848
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5605059, 2.5681005
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0388985, 2.0350637
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4772506, 2.4757848
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3160777, 2.3193221
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7998695, 1.8048511
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1044536, 2.1004667
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0904808, 3.0745721
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5100572, 1.5083876
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7451086, 2.7485285

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0572890, upper bound: 1.0475308
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523730, upper bound: 1.0524467
time: 6.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1060948, 2.1057701
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5630369, 2.5655694
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0385895, 2.0353732
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4757867, 2.4772482
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3194499, 2.3159499
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8016334, 1.8030870
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0954642, 2.1094561
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0848255, 3.0802274
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5061471, 1.5122981
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7493114, 2.7443256

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531099, upper bound: 1.0517107
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0481932, upper bound: 1.0566258
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1065454, 2.1041164
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5624390, 2.5633659
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0357285, 2.0363579
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4770389, 2.4758716
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3153195, 2.3196321
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8010721, 1.8019071
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1066756, 2.0950260
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0798998, 3.0788956
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5109198, 1.5062773
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7438011, 2.7490625

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573544, upper bound: 1.0474638
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524389, upper bound: 1.0523810
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1040602, 2.1066017
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5649700, 2.5608349
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0354195, 2.0366669
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4755759, 2.4773350
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3186922, 2.3162594
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8028359, 1.8001430
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0976872, 2.1040154
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0742435, 3.0845509
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5070093, 1.5101876
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7480040, 2.7448597

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531748, upper bound: 1.0516429
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0482598, upper bound: 1.0565596
time: 6.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0565600, upper bound: 1.0482600
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0516430, upper bound: 1.0531752
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0523810, upper bound: 1.0524387
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0474638, upper bound: 1.0573539
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0566255, upper bound: 1.0481933
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0517089, upper bound: 1.0531093
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0524466, upper bound: 1.0523730
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0475323, upper bound: 1.0572888
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0572890, upper bound: 1.0475308
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0523730, upper bound: 1.0524467
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0531099, upper bound: 1.0517107
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0481932, upper bound: 1.0566258
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0573544, upper bound: 1.0474638
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0524389, upper bound: 1.0523810
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0531748, upper bound: 1.0516429
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.42
Output dim: 0, lower bound: -1.0482598, upper bound: 1.0565596

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0863810, 2.0826125
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5578413, 2.5642152
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0310183, 2.0310969
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4884877, 2.4833794
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3036146, 2.3062248
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8042693, 1.8078980
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0952296, 2.0916910
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0696707, 3.0555172
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5015893, 1.4970779
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7437210, 2.7439957

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565563, upper bound: 1.0475450
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516684, upper bound: 1.0475477
time: 5.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0821085, 2.0868850
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5590839, 2.5629711
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0322733, 2.0298414
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4864249, 2.4854426
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3076849, 2.3021545
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8049607, 1.8072066
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0923743, 2.0945468
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0738707, 3.0513163
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4997249, 1.4989429
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7445230, 2.7431927

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516393, upper bound: 1.0524607
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0467517, upper bound: 1.0524632
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0838957, 2.0850978
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5603724, 2.5616846
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0307093, 2.0314059
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4870248, 2.4848428
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3069873, 2.3028522
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8060331, 1.8061342
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0862412, 2.1006799
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0640154, 3.0611725
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4976792, 1.5009885
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7479239, 2.7397928

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516686, upper bound: 1.0475474
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516659, upper bound: 1.0524352
time: 6.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0796232, 2.0893703
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5616150, 2.5604401
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0319643, 2.0301509
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4849610, 2.4869061
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3110571, 2.2987823
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8067245, 1.8054428
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0833850, 2.1035357
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0682154, 3.0569715
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4958138, 1.5028534
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7487259, 2.7389898

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0467520, upper bound: 1.0524635
time: 6.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0467492, upper bound: 1.0573509
time: 7.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0843463, 2.0834451
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5597744, 2.5594811
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0278482, 2.0323906
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4882770, 2.4834666
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3028569, 2.3065343
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8054714, 1.8049541
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0974522, 2.0862503
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0590897, 3.0598412
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5024524, 1.4949677
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7424135, 2.7445297

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0566218, upper bound: 1.0474795
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517338, upper bound: 1.0474820
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0800738, 2.0877175
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5610180, 2.5582371
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0291033, 2.0311346
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4862132, 2.4855294
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3069267, 2.3024640
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8061628, 1.8042626
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0945964, 2.0891066
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0632896, 3.0556402
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5005879, 1.4968324
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7432156, 2.7437272

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517052, upper bound: 1.0523950
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0468174, upper bound: 1.0523976
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0818610, 2.0859303
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5623055, 2.5569501
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0275393, 2.0326996
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4868131, 2.4849296
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3062291, 2.3031616
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8072352, 1.8031902
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0884633, 2.0952392
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0534334, 3.0654964
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4985423, 1.4988780
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7466164, 2.7403269

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517341, upper bound: 1.0474818
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517313, upper bound: 1.0523693
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0775890, 2.0902028
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5635490, 2.5557060
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0287943, 2.0314441
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4847503, 2.4869928
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3102994, 2.2990918
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8079267, 1.8024988
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0856071, 2.0980949
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0576344, 3.0612955
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4966769, 1.5007429
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7474184, 2.7395244

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0468177, upper bound: 1.0523974
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0468149, upper bound: 1.0572853
time: 5.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0871558, 2.0775886
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5557060, 2.5645452
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0313740, 2.0287938
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4871173, 2.4835887
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2995400, 2.3068547
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8024988, 1.8081713
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0956678, 2.0888257
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0675526, 3.0558443
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5007434, 1.4972088
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7402973, 2.7445202

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0572853, upper bound: 1.0468151
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523974, upper bound: 1.0468176
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0828834, 2.0818610
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5569506, 2.5633011
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0326290, 2.0275393
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4850545, 2.4856520
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3036103, 2.3027844
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8031902, 1.8074799
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0928121, 2.0916815
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0717535, 3.0516434
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4988780, 1.4990737
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7411003, 2.7437172

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523693, upper bound: 1.0517314
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474818, upper bound: 1.0517341
time: 7.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0846705, 2.0800738
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5582371, 2.5620141
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0310650, 2.0291028
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4856544, 2.4850521
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3029122, 2.3034821
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8042626, 1.8064072
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0866790, 2.0978146
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0618973, 3.0614996
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4968324, 1.5011194
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7445011, 2.7403173

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523976, upper bound: 1.0468176
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523949, upper bound: 1.0517052
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0803981, 2.0843463
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5594807, 2.5607700
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0323191, 2.0278487
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4835906, 2.4871154
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3069825, 2.2994123
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8049541, 1.8057158
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0838227, 2.1006708
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0660982, 3.0572991
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4949679, 1.5029843
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7453032, 2.7395144

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474820, upper bound: 1.0517340
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474792, upper bound: 1.0566219
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0851212, 2.0784211
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5576401, 2.5598111
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0282040, 2.0300875
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4869065, 2.4836760
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2987823, 2.3071647
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8037009, 1.8052273
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0978909, 2.0833850
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0569715, 3.0601683
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5016065, 1.4950984
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7389898, 2.7450562

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573508, upper bound: 1.0467492
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524628, upper bound: 1.0467519
time: 6.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0808487, 2.0826936
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5588837, 2.5585666
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0294590, 2.0288324
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4848428, 2.4857392
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3028522, 2.3030944
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8043923, 1.8045359
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0950346, 2.0862408
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0611725, 3.0559678
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4997411, 1.4969633
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7397928, 2.7442536

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524352, upper bound: 1.0516659
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0475474, upper bound: 1.0516686
time: 5.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0565563, upper bound: 1.0475450
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0516684, upper bound: 1.0475477
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0516393, upper bound: 1.0524607
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0467517, upper bound: 1.0524632
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0516686, upper bound: 1.0475474
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0516659, upper bound: 1.0524352
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0467520, upper bound: 1.0524635
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0467492, upper bound: 1.0573509
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0566218, upper bound: 1.0474795
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0517338, upper bound: 1.0474820
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0517052, upper bound: 1.0523950
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0468174, upper bound: 1.0523976
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0517341, upper bound: 1.0474818
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0517313, upper bound: 1.0523693
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0468177, upper bound: 1.0523974
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0468149, upper bound: 1.0572853
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0572853, upper bound: 1.0468151
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0523974, upper bound: 1.0468176
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0523693, upper bound: 1.0517314
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0474818, upper bound: 1.0517341
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0523976, upper bound: 1.0468176
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0523949, upper bound: 1.0517052
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0474820, upper bound: 1.0517340
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0474792, upper bound: 1.0566219
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0573508, upper bound: 1.0467492
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0524628, upper bound: 1.0467519
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0524352, upper bound: 1.0516659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.31
Output dim: 0, lower bound: -1.0475474, upper bound: 1.0516686
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 0, lower bound: -1.0531748, upper bound: 1.0516429
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.31
Output dim: 0, lower bound: -1.0482598, upper bound: 1.0565596
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.1192193031311035
rel_dist={0: [-1.0573893980579818, 1.057389512159725]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9087012, upper bound: 0.9094478
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094485, upper bound: 0.9087007
time: 6.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.62
Output dim: 0, lower bound: -0.9087012, upper bound: 0.9094478
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.62
Output dim: 0, lower bound: -0.9094485, upper bound: 0.9087007

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0331540, 2.0369225
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4385061, 2.4369040
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9285555, 1.9302819
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3636894, 2.3626623
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2105193, 2.2074633
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7188377, 1.7175100
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0721130, 2.0742621
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9664392, 2.9648509
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4439604, 1.4433253
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7008905, 2.6983228

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086679, upper bound: 0.9094472
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9087007, upper bound: 0.9094144
time: 6.09 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0339293, 2.0331545
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4369040, 2.4372334
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9289112, 1.9285553
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3626614, 2.3628716
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2074633, 2.2080922
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7175097, 1.7177835
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0725513, 2.0721130
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9648514, 2.9651775
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4433253, 1.4434559
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6983232, 2.6988478

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094151, upper bound: 0.9087001
time: 6.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094479, upper bound: 0.9086673
time: 6.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 0, lower bound: -0.9086679, upper bound: 0.9094472
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 0, lower bound: -0.9087007, upper bound: 0.9094144
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 0, lower bound: -0.9094151, upper bound: 0.9087001
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.53
Output dim: 0, lower bound: -0.9094479, upper bound: 0.9086673

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0338492, 2.0360909
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4365721, 2.4385209
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9296393, 1.9289892
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3637619, 2.3625760
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2107778, 2.2071533
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7176356, 1.7185154
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0698905, 2.0761199
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9700499, 2.9605260
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4430974, 1.4440455
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7013369, 2.6977882

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086656, upper bound: 0.9064469
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9056674, upper bound: 0.9094451
time: 6.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0323229, 2.0369225
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4385061, 2.4349704
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9272618, 1.9302819
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3636036, 2.3626623
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2102094, 2.2074633
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7188377, 1.7163072
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0721130, 2.0720391
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9621143, 2.9648509
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4439604, 1.4424627
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7003565, 2.6983228

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086985, upper bound: 0.9064142
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9057002, upper bound: 0.9094128
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0346241, 2.0323229
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4349699, 2.4388504
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9299970, 1.9272625
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3627338, 2.3627853
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2077217, 2.2077827
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7163072, 1.7187886
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0703297, 2.0739713
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9684620, 2.9608536
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4424622, 1.4441762
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6987696, 2.6983142

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094129, upper bound: 0.9056996
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9064147, upper bound: 0.9086976
time: 6.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0330982, 2.0331545
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4369040, 2.4352999
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9276185, 1.9285553
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3625755, 2.3628716
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2071533, 2.2080922
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7175097, 1.7165809
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0725513, 2.0698905
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9605255, 2.9651775
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4433253, 1.4425933
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6977882, 2.6988478

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094457, upper bound: 0.9056669
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9064475, upper bound: 0.9086648
time: 6.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9086656, upper bound: 0.9064469
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9056674, upper bound: 0.9094451
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9086985, upper bound: 0.9064142
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9057002, upper bound: 0.9094128
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9094129, upper bound: 0.9056996
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9064147, upper bound: 0.9086976
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9094457, upper bound: 0.9056669
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.76
Output dim: 0, lower bound: -0.9064475, upper bound: 0.9086648

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0213852, 2.0217633
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4213743, 2.4252214
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9298697, 1.9289875
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3709855, 2.3687019
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1928840, 2.1917892
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7124286, 1.7146316
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0227237, 2.0222116
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9488344, 2.9350691
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4225843, 1.4205990
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6760855, 2.6756892

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086576, upper bound: 0.9027785
time: 7.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9049964, upper bound: 0.9064391
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0195212, 2.0236273
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4232721, 2.4233232
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9296389, 1.9292192
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3698878, 2.3697996
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1954131, 2.1892595
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7137518, 1.7133086
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0159822, 2.0289531
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9445934, 2.9393106
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4196508, 1.4235318
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6792374, 2.6725368

Time for backsubstitution: 14.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9056594, upper bound: 0.9057768
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9019981, upper bound: 0.9094372
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0198593, 2.0225945
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4233084, 2.4216709
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9274931, 1.9302812
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3708272, 2.3687887
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1923156, 2.1920991
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7136312, 1.7124236
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0249457, 2.0181308
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9408989, 2.9393935
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4234469, 1.4190161
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6751041, 2.6762233

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086904, upper bound: 0.9027457
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050292, upper bound: 0.9064063
time: 7.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0179954, 2.0244584
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4252062, 2.4197726
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9272604, 1.9305134
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3697295, 2.3698864
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1948447, 2.1895695
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7149544, 1.7111006
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0182042, 2.0248728
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9366570, 2.9436350
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4205139, 1.4219489
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6782570, 2.6730714

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9056922, upper bound: 0.9057441
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9020310, upper bound: 0.9094045
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0221605, 2.0179954
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4197721, 2.4255509
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9302263, 1.9272609
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3699584, 2.3689113
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1898279, 2.1924191
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7111006, 1.7149050
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0231624, 2.0200624
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9472466, 2.9353971
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4219491, 1.4207296
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6735172, 2.6762137

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094050, upper bound: 0.9020328
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057445, upper bound: 0.9056941
time: 5.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0202966, 2.0198593
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4216709, 2.4236526
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9299936, 1.9274926
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3688607, 2.3700085
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1923575, 2.1898899
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7124233, 1.7135820
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0164204, 2.0268044
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9430046, 2.9396381
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4190166, 1.4236624
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6766701, 2.6730614

Time for backsubstitution: 14.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9064068, upper bound: 0.9050284
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9027463, upper bound: 0.9086898
time: 7.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0206347, 2.0188265
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4217062, 2.4220004
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9278479, 1.9285545
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3697991, 2.3689981
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1892595, 2.1927290
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7123032, 1.7126970
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0253844, 2.0159822
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9393101, 2.9397202
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4228117, 1.4191468
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6725368, 2.6767483

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094378, upper bound: 0.9019975
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057773, upper bound: 0.9056589
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0187707, 2.0206904
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4236050, 2.4201021
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9276161, 1.9287868
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3687015, 2.3700953
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1917892, 2.1901994
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7136259, 1.7113740
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0186429, 2.0227242
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9350691, 2.9439616
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4198787, 1.4220796
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6756887, 2.6735959

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9064396, upper bound: 0.9049958
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9027791, upper bound: 0.9086579
time: 6.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 28.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9086576, upper bound: 0.9027785
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9049964, upper bound: 0.9064391
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9056594, upper bound: 0.9057768
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9019981, upper bound: 0.9094372
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9086904, upper bound: 0.9027457
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9050292, upper bound: 0.9064063
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9056922, upper bound: 0.9057441
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9020310, upper bound: 0.9094045
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9094050, upper bound: 0.9020328
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9057445, upper bound: 0.9056941
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9064068, upper bound: 0.9050284
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9027463, upper bound: 0.9086898
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9094378, upper bound: 0.9019975
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9057773, upper bound: 0.9056589
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9064396, upper bound: 0.9049958
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.66
Output dim: 0, lower bound: -0.9027791, upper bound: 0.9086579

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9988933, 1.9960670
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4165745, 2.4213552
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9223452, 1.9224043
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3803372, 2.3765054
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1763463, 2.1783042
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7150578, 1.7177792
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0132246, 2.0105705
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9259071, 2.9152918
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4128041, 1.4094198
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6712742, 2.6714802

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086550, upper bound: 0.9021518
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050345, upper bound: 0.9021537
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9938250, 2.0011353
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4194050, 2.4185238
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9230547, 1.9216948
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3776917, 2.3791509
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1819282, 2.1727219
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7168989, 1.7159376
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0043411, 2.0194540
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9248161, 2.9163828
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4084725, 1.4137514
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6750278, 2.6677260

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9013736, upper bound: 0.9058139
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9013715, upper bound: 0.9094346
time: 6.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9973674, 1.9968996
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4185085, 2.4178042
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9199686, 1.9236979
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3801789, 2.3765926
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1757779, 2.1786137
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7162600, 1.7155712
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0154467, 2.0064898
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9179707, 2.9196157
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4136662, 1.4078372
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6702938, 2.6720142

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086879, upper bound: 0.9021190
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050673, upper bound: 0.9021210
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9922991, 2.0019679
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4213390, 2.4149733
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9206772, 1.9229879
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3775334, 2.3792377
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1813598, 2.1730313
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7181010, 1.7137299
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0065632, 2.0153737
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9168797, 2.9207067
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4093356, 1.4121685
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6740475, 2.6682606

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9014064, upper bound: 0.9057812
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9014044, upper bound: 0.9094018
time: 6.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9996681, 1.9922991
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4149733, 2.4216847
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9227009, 1.9206772
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3793092, 2.3767147
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1732907, 2.1789341
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7137299, 1.7180524
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0136623, 2.0084214
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9243183, 2.9156189
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4121690, 1.4095509
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6687069, 2.6720047

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094024, upper bound: 0.9014039
time: 8.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057819, upper bound: 0.9014058
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9945998, 1.9973674
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4178047, 2.4188533
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9234104, 1.9199681
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3766637, 2.3793602
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1788726, 2.1733518
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7155714, 1.7162108
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0047789, 2.0173054
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9232273, 2.9167099
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4078374, 1.4138823
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6724606, 2.6682506

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9021217, upper bound: 0.9050692
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9021197, upper bound: 0.9086897
time: 8.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9981418, 1.9931316
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4169064, 2.4181342
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9203234, 1.9219704
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3791509, 2.3768020
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1727223, 2.1792440
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7149320, 1.7158444
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0158854, 2.0043411
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9163828, 2.9199433
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4130321, 1.4079678
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6677265, 2.6725407

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094352, upper bound: 0.9013715
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9058147, upper bound: 0.9013729
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9930739, 1.9981999
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4197369, 2.4153028
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9210329, 1.9212618
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3765054, 2.3794475
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1783042, 2.1736617
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7167735, 1.7140028
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0070019, 2.0132246
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9152918, 2.9210339
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4087005, 1.4122994
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6714802, 2.6687870

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9021545, upper bound: 0.9050338
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9021525, upper bound: 0.9086543
time: 6.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9086550, upper bound: 0.9021518
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9050345, upper bound: 0.9021537
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9013736, upper bound: 0.9058139
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9013715, upper bound: 0.9094346
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9086879, upper bound: 0.9021190
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9050673, upper bound: 0.9021210
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9014064, upper bound: 0.9057812
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9014044, upper bound: 0.9094018
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9094024, upper bound: 0.9014039
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9057819, upper bound: 0.9014058
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9021217, upper bound: 0.9050692
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9021197, upper bound: 0.9086897
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9094352, upper bound: 0.9013715
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9058147, upper bound: 0.9013729
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9021545, upper bound: 0.9050338
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.15
Output dim: 0, lower bound: -0.9021525, upper bound: 0.9086543

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9953432, 1.9912534
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4119024, 2.4179087
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9224796, 1.9215369
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3790565, 2.3755608
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1716652, 2.1748471
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7126951, 1.7164614
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0025949, 1.9961629
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9272971, 2.9152794
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4054792, 1.3994966
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6673813, 2.6686058

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9084438, upper bound: 0.9021513
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086546, upper bound: 0.9019405
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9890113, 1.9975848
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4159584, 2.4138517
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9221869, 1.9218287
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3767476, 2.3778701
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1784716, 2.1680408
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7155814, 1.7135749
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -1.9899335, 2.0088248
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9248033, 2.9177732
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.3985488, 1.4064269
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6721535, 2.6638336

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9011603, upper bound: 0.9094341
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9013711, upper bound: 0.9092233
time: 5.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9938169, 1.9920850
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4138374, 2.4143577
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9201031, 1.9228306
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3788981, 2.3756475
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1710968, 2.1751566
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7138968, 1.7142534
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0048170, 1.9920826
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9193616, 2.9196029
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4063423, 1.3979139
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6664009, 2.6691399

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.24 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9084766, upper bound: 0.9021189
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086874, upper bound: 0.9019078
time: 5.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9874854, 1.9984164
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4178925, 2.4103012
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9198093, 1.9231229
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3765893, 2.3779569
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1779032, 2.1683502
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7167830, 1.7113669
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -1.9921556, 2.0047441
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9168677, 2.9220967
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.3994119, 1.4048440
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6711731, 2.6643677

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9011931, upper bound: 0.9094015
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9014039, upper bound: 0.9091908
time: 12.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9961181, 1.9874854
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4103012, 2.4182382
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9228354, 1.9198093
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3780284, 2.3757701
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1686091, 2.1754770
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7113671, 1.7167344
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0030327, 1.9940143
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9257092, 2.9156065
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4048440, 1.3996277
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6648140, 2.6691303

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.23 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9091912, upper bound: 0.9014032
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094019, upper bound: 0.9011930
time: 5.65 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 26.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9084438, upper bound: 0.9021513
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9086546, upper bound: 0.9019405
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9011603, upper bound: 0.9094341
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9013711, upper bound: 0.9092233
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9084766, upper bound: 0.9021189
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9086874, upper bound: 0.9019078
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9011931, upper bound: 0.9094015
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9014039, upper bound: 0.9091908
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9091912, upper bound: 0.9014032
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 26.25
Output dim: 0, lower bound: -0.9094019, upper bound: 0.9011930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.25
Output dim: 0, lower bound: -0.9021197, upper bound: 0.9086897
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.25
Output dim: 0, lower bound: -0.9094352, upper bound: 0.9013715
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.25
Output dim: 0, lower bound: -0.9021525, upper bound: 0.9086543
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.0339293479919434
rel_dist={0: [-0.9094671207244254, 0.9094665870331546]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2419.93 seconds
